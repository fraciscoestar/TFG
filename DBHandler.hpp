#ifndef DBHANDLER_H
#define DBHANDLER_H

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "sqlite3.h"

#define FILTER_HPF 0
#define FILTER_HFE 1

#define MINMATCHES 65

using namespace std;
using namespace cv;

class DBHandler
{
public:

    static bool Login(string* username);

    static bool Register(string username);

    static tuple<string, vector<KeyPoint>, Mat> FindBestMatch(tuple<vector<KeyPoint>, Mat> features);

    static tuple<string, vector<KeyPoint>, Mat> ReadEntry(int id, sqlite3 *e_db = NULL);

    static int WriteEntry(tuple<vector<KeyPoint>, Mat> features, string name);

    static char* EncodeF32Image(Mat& img);

    static Mat DecodeKazeDescriptor(vector<char> &buffer, int nKeypoints);

    static tuple<vector<KeyPoint>, Mat> KAZEDetector(Mat& src);

    static tuple<Mat, int> FLANNMatcher(tuple<vector<KeyPoint>, Mat> m1, tuple<vector<KeyPoint>, Mat> m2, Mat imgA = Mat(), Mat imgB = Mat());

    static void PreprocessImage(Mat &src, Mat &dst);

    static void CGF(Mat &src, Mat &dst);

    static Mat DFTModule(Mat src[], bool shift);

    static void HPF(Mat& src, Mat& dst, uint8_t filterType);

    static void FFTShift(const Mat& src, Mat &dst);

private:
    static const int d0hfe = 10, d0hpf = 40, k1hfe = 5300, k2hfe = 7327, k1hpf = 3050, k2hpf = 5463, sig = 5, dF=436;
    DBHandler() {}
};

#endif 