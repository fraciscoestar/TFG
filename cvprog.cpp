#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <wiringPi.h>
#include <unistd.h>
#include <signal.h>
#include "sqlite3.h"
#include "DBHandler.hpp"

#define FILTER_HPF 0
#define FILTER_HFE 1

using namespace std;
using namespace cv;

Mat TakeImage();
bool LoginRP4(string* username);
bool RegisterRP4(string username);

void handler(int signal, siginfo_t *data, void*) { }

int d0hfe = 10, d0hpf = 40, k1hfe = 5300, k2hfe = 7327, k1hpf = 3050, k2hpf = 5463, sig = 5, dF=436;

int main(int argc, char const *argv[])
{
    wiringPiSetup();
    pinMode(4, OUTPUT);
    pinMode(5, OUTPUT);

    // Signals ///////////////////////
    struct sigaction sa;
    siginfo_t info;
    sigset_t ss;

    sa.sa_sigaction = handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGRTMIN, &sa, NULL);

    sigemptyset(&ss);
    sigaddset(&ss, SIGRTMIN);
    sigprocmask(SIG_BLOCK, &ss, NULL);
    //////////////////////////////////

    if (argc > 1)
    {
        int action = atoi(argv[1]);

        switch (action)
        {
            case 0: // Login
                if (argc > 2)
                {
                    int pid = atoi(argv[2]);
                    union sigval value;
                    string username;

                    value.sival_int = (int)LoginRP4(&username);

                    if(value.sival_int != 0)
                    {
                        value.sival_int = username.length();
                    }

                    sigqueue(pid, SIGRTMIN, value); // Sends 0 if user is not found in db, size of name if successful. 
                    sigwaitinfo(&ss, &info);       

                    for (int i = 0; i < value.sival_int; i++) // Send characters of username
                    {
                        value.sival_int = (int)username.at(i);
                        sigqueue(pid, SIGRTMIN, value);
                        sigwaitinfo(&ss, &info);   
                    }                   

                    return 0;
                }
                break;

            case 1: // Register
                if (argc > 3)
                {
                    int pid = atoi(argv[2]);
                    string username = string(argv[3]);
                    union sigval value;

                    value.sival_int = (int)RegisterRP4(username);
                    sigqueue(pid, SIGRTMIN, value); // Sends 0 if user already registered, 1 if successful.   
                    return 0;                
                }
                break;

            default:
                break;
        }

        return 1;
    }

    // No arguments -> show vascularity
    Mat img, img2;
    img = TakeImage();
    DBHandler::PreprocessImage(img, img2);
    imshow("img", img);
    imshow("Vascularity", img2);

    waitKey(0);
    destroyAllWindows();
    
    return 0;
}

bool LoginRP4(string* username)
    {
    Mat img, imgPre;
    img = TakeImage();
    DBHandler::PreprocessImage(img, imgPre);
    tuple<vector<KeyPoint>, Mat> features = DBHandler::KAZEDetector(imgPre);
    tuple<string, vector<KeyPoint>, Mat> bestMatch = DBHandler::FindBestMatch(features);

    if (get<0>(bestMatch) == "")
        return false;
    else
    {
        *username = get<0>(bestMatch);
        return true;
    }
}

bool RegisterRP4(string username)
{
    Mat img, imgPre;

    img = TakeImage();
    DBHandler::PreprocessImage(img, imgPre);
    tuple<vector<KeyPoint>, Mat> features = DBHandler::KAZEDetector(imgPre);
    return !DBHandler::WriteEntry(features, username);
}

Mat TakeImage()
{
    VideoCapture vc(0);
    Mat img, img2;

    digitalWrite(4, HIGH);
    digitalWrite(5, HIGH);

    vc.read(img);
    cvtColor(img, img, COLOR_BGR2GRAY);

    Rect fingerRegion = Rect(255, 0, 465-255, img.rows-40);
    img2 = img(fingerRegion);

    digitalWrite(4, LOW);
    digitalWrite(5, LOW);
    
    return img2.clone();
}