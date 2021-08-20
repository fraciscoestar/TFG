#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <wiringPi.h>

using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
    Mat img;
    VideoCapture vc(0);

    while (true)
    {
        // Activate Lights ////////////////
        wiringPiSetup();
        pinMode(4, OUTPUT);
        pinMode(5, OUTPUT);

        digitalWrite(4, HIGH); //980nm
        digitalWrite(5, HIGH); //850nm
        ///////////////////////////////////

        // Take image /////////////////////
        vc.read(img);
        imshow("Img", img);
        ///////////////////////////////////

        // Deactivate Lights //////////////
        digitalWrite(4, LOW);
        digitalWrite(5, LOW);
        ///////////////////////////////////
        
        waitKey(1);
    }

    return 0;
}