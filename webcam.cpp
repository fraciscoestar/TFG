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
        vc.read(img);
        imshow("Img", img);

        waitKey(1);
    }

    return 0;
}