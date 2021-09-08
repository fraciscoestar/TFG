#include <iostream>
#include <fstream>
#include <iterator>
#include <signal.h>
#include <unistd.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "sqlite3.h"
#include "DBHandler.hpp"

using namespace std;
using namespace cv;

void test();
void PerformTest(Mat &src);
tuple<vector<KeyPoint>, Mat> SURFDetector(Mat &src);
tuple<vector<KeyPoint>, Mat> SIFTDetector(Mat& src);
tuple<vector<KeyPoint>, Mat> ORBDetector(Mat& src);
tuple<Mat, int> BruteForceMatcher(tuple<vector<KeyPoint>, Mat> m1, tuple<vector<KeyPoint>, Mat> m2, Mat imgA = Mat(), Mat imgB = Mat());
void TestMatchers();

void handler(int signal, siginfo_t* data, void*) { }

int main(int argc, char const *argv[])
{
    // Signals ////////////////////////
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
    ///////////////////////////////////
    
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

                    value.sival_int = (int)DBHandler::Login(&username);

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

                    exit(0);
                }
                break;

            case 1: // Register
                if (argc > 3)
                {
                    int pid = atoi(argv[2]);
                    string username = string(argv[3]);
                    union sigval value;

                    value.sival_int = (int)DBHandler::Register(username);
                    sigqueue(pid, SIGRTMIN, value); // Sends 0 if user already registered, 1 if successful.   
                    exit(0);                
                }
                break;

            default:
                break;
        }

        exit(1);
    }

    // No arguments -> perform test

    Mat img;
    
    // Take images /////////////////////
    img = imread("original4.png", ImreadModes::IMREAD_GRAYSCALE);
    imshow("Img", img);
    ///////////////////////////////////

    PerformTest(img);

    waitKey(0);
    return 0;
}

void TestMatchers()
{
    Mat img, img2, imga, imgb, resA, resB;

    // Take images /////////////////////
    img = imread("original.png", ImreadModes::IMREAD_GRAYSCALE);
    img2 = imread("original4.png", ImreadModes::IMREAD_GRAYSCALE);
    imshow("Img", img);
    imshow("Img2", img2);
    ///////////////////////////////////
    
    // Preprocess /////////////////////
    DBHandler::PreprocessImage(img, imga);
    DBHandler::PreprocessImage(img2, imgb);
    ///////////////////////////////////

    // SURF ///////////////////////////
    tuple<vector<KeyPoint>, Mat> surf1 = SURFDetector(imga);
    tuple<vector<KeyPoint>, Mat> surf2 = SURFDetector(imgb);
    ///////////////////////////////////

    // SIFT ///////////////////////////
    tuple<vector<KeyPoint>, Mat> sift1 = SURFDetector(imga);
    tuple<vector<KeyPoint>, Mat> sift2 = SURFDetector(imgb);
    ///////////////////////////////////

    // ORB ////////////////////////////
    tuple<vector<KeyPoint>, Mat> orb1 = ORBDetector(imga);
    tuple<vector<KeyPoint>, Mat> orb2 = ORBDetector(imgb);
    ///////////////////////////////////

    // KAZE ///////////////////////////
    tuple<vector<KeyPoint>, Mat> kaze1 = DBHandler::KAZEDetector(imga);
    tuple<vector<KeyPoint>, Mat> kaze2 = DBHandler::KAZEDetector(imgb);
    ///////////////////////////////////

    // Match Features /////////////////
    tuple<Mat, int> matchesSURF = DBHandler::FLANNMatcher(surf1, surf2, imga, imgb);
    imshow("SURF Matches", get<0>(matchesSURF));

    tuple<Mat, int> matchesSIFT = DBHandler::FLANNMatcher(sift1, sift2, imga, imgb);
    imshow("SIFT Matches", get<0>(matchesSIFT));

    tuple<Mat, int> matchesORB = BruteForceMatcher(orb1, orb2, imga, imgb);
    imshow("ORB Matches", get<0>(matchesORB));

    tuple<Mat, int> matchesKAZE = DBHandler::FLANNMatcher(kaze1, kaze2, imga, imgb);
    imshow("KAZE Matches", get<0>(matchesKAZE));
    ///////////////////////////////////
}

tuple<vector<KeyPoint>, Mat> SURFDetector(Mat& src)
{
    Mat img = src.clone();

    // SURF ///////////////////////////
    Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(5000);

    vector<KeyPoint> keypoints;
    Mat descriptors;
        
    detector->detectAndCompute(img, noArray(), keypoints, descriptors);
    return make_tuple(keypoints, descriptors);
    ///////////////////////////////////
}

tuple<vector<KeyPoint>, Mat> SIFTDetector(Mat& src)
{
    Mat img = src.clone();

    // SURF ///////////////////////////
    Ptr<SIFT> detector = SIFT::create(1000);

    vector<KeyPoint> keypoints;
    Mat descriptors;
        
    detector->detectAndCompute(img, noArray(), keypoints, descriptors);
    return make_tuple(keypoints, descriptors);
    ///////////////////////////////////
}

tuple<vector<KeyPoint>, Mat> ORBDetector(Mat& src)
{
    Mat img = src.clone();

    // SURF ///////////////////////////
    Ptr<ORB> detector = ORB::create(1000);

    vector<KeyPoint> keypoints;
    Mat descriptors;
        
    detector->detectAndCompute(img, noArray(), keypoints, descriptors);
    return make_tuple(keypoints, descriptors);
    ///////////////////////////////////
}

tuple<Mat, int> BruteForceMatcher(tuple<vector<KeyPoint>, Mat> m1, tuple<vector<KeyPoint>, Mat> m2, Mat imgA, Mat imgB)
{
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING);
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(get<1>(m1), get<1>(m2), knn_matches, 2);

    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    float maxInc = 0.34f;
    std::vector<DMatch> goodest_matches;

    for (size_t i = 0; i < good_matches.size(); i++)
    {
        int idx1 = good_matches[i].trainIdx;
        int idx2 = good_matches[i].queryIdx;

        const KeyPoint &kp1 = get<0>(m2)[idx1], &kp2 = get<0>(m1)[idx2];
        Point2f p1 = kp1.pt;
        Point2f p2 = kp2.pt;
        Point2f triangle = Point2f(std::abs(p2.x - p1.x), std::abs(p2.y - p1.y));

        float angle = std::atan2(triangle.y, triangle.x);

        if (std::abs(angle) < maxInc)
        {
            goodest_matches.push_back(good_matches[i]);
        }
    }
    

    cout << "Good matches: " << to_string(good_matches.size()) << endl;
    cout << "Goodest matches: " << to_string(goodest_matches.size()) << endl;

    if (imgA.rows > 1 && imgB.rows > 1)
    {
        Mat img_matches;
        drawMatches(imgA.clone(), get<0>(m1), imgB.clone(), get<0>(m2), goodest_matches, img_matches, Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        return make_tuple(img_matches.clone(), good_matches.size());
    }
    else
    {
        return make_tuple(Mat(), good_matches.size());
    }
}

void PerformTest(Mat& src)
{
    Mat img, img2, img3, img4, img5, img6, img7, img8, img9, img10;

    img2 = src.clone();
    // Umbralizar /////////////////////
    threshold(img2, img3, 60, 255, THRESH_BINARY);

    Mat framed = Mat::zeros(Size2d(img3.cols + 2, img3.rows + 2), CV_8U);
    Rect r = Rect2d(1, 1, img3.cols, img3.rows);
    img3.copyTo(framed(r));

    Canny(framed, img4, 10, 50);
    
    Mat kernel = getStructuringElement(MORPH_RECT, Size2d(5,5));
    dilate(img4, img4, kernel);

    vector<vector<Point>> contours;
    findContours(img4, contours, RETR_TREE, CHAIN_APPROX_NONE);

    Mat contoursImg = Mat::zeros(img4.size(), CV_8U);
    double maxArea = 0;
    int mIdx = -1;
    for (int i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if(area > maxArea)
        {
            maxArea = area;
            mIdx = i;
        }
    }

    if(mIdx >= 0)
        drawContours(contoursImg, contours, mIdx, Scalar::all(255), FILLED);


    bitwise_and(img2, img2, img5, contoursImg(r));    
    ///////////////////////////////////

    // CLAHE //////////////////////////
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(6);

    clahe->apply(img5, img5);
    imshow("CLAHE", img5);
    ///////////////////////////////////

    Mat imga, imgb, imgc, imgd, imge;

    // HFE //////////////////////
    DBHandler::HPF(img5, img6, FILTER_HFE);
    resize(img6, img8, img5.size());
    bitwise_and(img8, img8, imga, contoursImg(r));
    clahe->apply(imga, imga);
    imshow("HFE", imga);
    ///////////////////////////////////   

    // HPF //////////////////////
    DBHandler::HPF(img5, img6, FILTER_HPF);
    resize(img6, img8, img5.size());
    bitwise_and(img8, img8, imgb, contoursImg(r));
    clahe->apply(imgb, imgb);
    imshow("HPF", imgb);
    ///////////////////////////////////   

    // HPF + HFE //////////////////////
    DBHandler::HPF(img5, img6, FILTER_HPF);
    DBHandler::HPF(img6, img7, FILTER_HFE);        
    resize(img7, img8, img5.size());
    bitwise_and(img8, img8, imgc, contoursImg(r));
    clahe->apply(imgc, imgc);
    imshow("HPF + HFE", imgc);
    ///////////////////////////////////   

    // HFE + HPF //////////////////////
    DBHandler::HPF(img5, img6, FILTER_HFE);
    DBHandler::HPF(img6, img7, FILTER_HPF);
    resize(img7, img8, img5.size());
    bitwise_and(img8, img8, imgd, contoursImg(r));
    clahe->apply(imgd, imgd);
    imshow("HFE + HPF", imgd);
    ///////////////////////////////////   

    // CGF ////////////////////////////
    DBHandler::CGF(imgd, imge);
    Mat mask;
    bitwise_not(contoursImg(r), mask);
    normalize(imge, imge, 1.0, 0.0, 4, -1, mask);
    imge.convertTo(imge, CV_8U);
    threshold(imge, imge, 0, 255, THRESH_BINARY);
        
    kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
    dilate(imge, imge, kernel);
    erode(imge, imge, kernel);

    kernel = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
    erode(imge, imge, kernel);
    dilate(imge, imge, kernel);
        
    imshow("gabor filtered", imge);   
    /////////////////////////////////// 

    // SURF ///////////////////////////
    tuple<vector<KeyPoint>, Mat> surf = SURFDetector(imge);
    Mat surfMat;
    drawKeypoints(imge, get<0>(surf), surfMat, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("SURF", surfMat);
    ///////////////////////////////////

    // SIFT ///////////////////////////
    tuple<vector<KeyPoint>, Mat> sift = SURFDetector(imge);
    Mat siftMat;
    drawKeypoints(imge, get<0>(surf), siftMat, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("SIFT", siftMat);
    ///////////////////////////////////

    // ORB ////////////////////////////
    tuple<vector<KeyPoint>, Mat> orb = ORBDetector(imge);
    Mat orbMat;
    drawKeypoints(imge, get<0>(surf), orbMat, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("ORB", orbMat);
    ///////////////////////////////////

    // KAZE ///////////////////////////
    tuple<vector<KeyPoint>, Mat> kaze = DBHandler::KAZEDetector(imge);
    Mat kazeMat;
    drawKeypoints(imge, get<0>(surf), kazeMat, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("KAZE", kazeMat);
    ///////////////////////////////////

    
}
 
