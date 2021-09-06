#include <iostream>
#include <fstream>
#include <iterator>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "sqlite3.h"

#define FILTER_HPF 0
#define FILTER_HFE 1

#define EMPTY_MAT Mat::zeros(Size2d(0,0), CV_8U)

using namespace std;
using namespace cv;

void HPF(Mat& src, Mat& dst, uint8_t filterType);
void FFTShift(const Mat& src, Mat &dst);
void test();
Mat DFTModule(Mat src[], bool shift);
void CGF(Mat &src, Mat &dst);
void PreprocessImage(Mat &src, Mat &dst);
void PerformTest(Mat &src, Mat &dst);
tuple<vector<KeyPoint>, Mat> SURFDetector(Mat &src);
tuple<vector<KeyPoint>, Mat> SIFTDetector(Mat& src);
tuple<vector<KeyPoint>, Mat> ORBDetector(Mat& src);
tuple<vector<KeyPoint>, Mat> KAZEDetector(Mat& src);
tuple<Mat, int> FLANNMatcher(tuple<vector<KeyPoint>, Mat> m1, tuple<vector<KeyPoint>, Mat> m2, Mat imgA = Mat(), Mat imgB = Mat());
tuple<Mat, int> BruteForceMatcher(tuple<vector<KeyPoint>, Mat> m1, tuple<vector<KeyPoint>, Mat> m2, Mat imgA = Mat(), Mat imgB = Mat());
int WriteEntry(tuple<vector<KeyPoint>, Mat> features, string name);
tuple<string, vector<KeyPoint>, Mat> ReadEntry(int id, sqlite3 *e_db = NULL);
void TestMatchers();
tuple<string, vector<KeyPoint>, Mat> FindBestMatch(tuple<vector<KeyPoint>, Mat> features);
char* EncodeF32Image(Mat& img);
Mat DecodeKazeDescriptor(vector<char> buffer, int nKeypoints);
void callbackButton(int state, void* userdata);

int d0hfe = 10, d0hpf = 40, k1hfe = 5300, k2hfe = 7327, k1hpf = 3050, k2hpf = 5463, sig = 5, dF=436;

int main(int argc, char const *argv[])
{
    Mat img, img2, imga, imgb, resA, resB;
    namedWindow("test", WINDOW_NORMAL);

    string nameb1 = "Login";
    string nameb2 = "Register";
    string nameb3 = "Exit";
    createButton(nameb1, callbackButton, &nameb1, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR, 0);
    createButton(nameb2, callbackButton, &nameb2, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR, 0);
    createButton(nameb3, callbackButton, &nameb3, QT_PUSH_BUTTON | QT_NEW_BUTTONBAR, 0);
    
    // Take images /////////////////////
    img = imread("original4.png", ImreadModes::IMREAD_GRAYSCALE);
    imshow("Img", img);
    ///////////////////////////////////
    
    // Preprocess /////////////////////
    PreprocessImage(img, imga);
    ///////////////////////////////////

    // KAZE ///////////////////////////
    tuple<vector<KeyPoint>, Mat> kaze = KAZEDetector(imga);
    ///////////////////////////////////

    //WriteEntry(kaze, "angela");
    tuple<string, vector<KeyPoint>, Mat> bestMatch = FindBestMatch(kaze);

    waitKey(0);
    return 0;
}

void callbackButton(int state, void* userdata)
{
    string button = *(string*)userdata;

    if (button == "Login")
    {
        std::cout << "Login" << std::endl;
    }
    else if (button == "Register")
    {
        std::cout << "Register" << std::endl;
    }
    else if (button == "Exit")
    {
        std::cout << "Exit" << std::endl;
    }
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
    PreprocessImage(img, imga);
    PreprocessImage(img2, imgb);
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
    tuple<vector<KeyPoint>, Mat> kaze1 = KAZEDetector(imga);
    tuple<vector<KeyPoint>, Mat> kaze2 = KAZEDetector(imgb);
    ///////////////////////////////////

    // Match Features /////////////////
    tuple<Mat, int> matchesSURF = FLANNMatcher(surf1, surf2, imga, imgb);
    imshow("SURF Matches", get<0>(matchesSURF));

    tuple<Mat, int> matchesSIFT = FLANNMatcher(sift1, sift2, imga, imgb);
    imshow("SIFT Matches", get<0>(matchesSIFT));

    tuple<Mat, int> matchesORB = BruteForceMatcher(orb1, orb2, imga, imgb);
    imshow("ORB Matches", get<0>(matchesORB));

    tuple<Mat, int> matchesKAZE = FLANNMatcher(kaze1, kaze2, imga, imgb);
    imshow("KAZE Matches", get<0>(matchesKAZE));
    ///////////////////////////////////
}

tuple<string, vector<KeyPoint>, Mat> FindBestMatch(tuple<vector<KeyPoint>, Mat> features)
{
    sqlite3 *db;
    int rc = sqlite3_open_v2("database.db", &db, SQLITE_OPEN_READONLY, NULL);
    if (rc != SQLITE_OK)
    {
        cerr << "DB open failed: " << sqlite3_errmsg(db) << endl;
        throw;
    }

    sqlite3_stmt* stmt = NULL;
    string query = "SELECT * FROM data";

    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        cerr << "Prepare failed: " << sqlite3_errmsg(db) << endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        throw;
    }
    
    tuple<string, vector<KeyPoint>, Mat> bestMatch;
    int maxMatches = 0;
    while (true)
    {
        rc = sqlite3_step(stmt);
        if (rc == SQLITE_ROW)
        {
            int id = sqlite3_column_int(stmt, 0);
            tuple<string, vector<KeyPoint>, Mat> entry = ReadEntry(id, db);

            // cout << "Name: " << get<0>(entry) << endl;
            // cout << "Keypoints: " << get<1>(entry).size() << endl;
            // imshow("Descriptor", get<2>(entry));

            // Match Features /////////////////
            tuple<Mat, int> matchKAZE = FLANNMatcher(features, make_tuple(get<1>(entry), get<2>(entry)));
            ///////////////////////////////////

            if (get<1>(matchKAZE) > maxMatches)
            {
                maxMatches = get<1>(matchKAZE);
                bestMatch = entry;
            }
        }
        else
            break;
    }
    
    if (maxMatches > 0) // Match found
    {
        cout << maxMatches << " found for user " << get<0>(bestMatch) << endl;
        return bestMatch;
    }
    else
        throw;
}

tuple<string, vector<KeyPoint>, Mat> ReadEntry(int id, sqlite3 *e_db)
{
    sqlite3 *db;
    int rc;

    if (e_db != NULL)
    {
        db = e_db;
    }
    else
    {
        rc = sqlite3_open_v2("database.db", &db, SQLITE_OPEN_READONLY, NULL);
        if (rc != SQLITE_OK)
        {
            cerr << "DB open failed: " << sqlite3_errmsg(db) << endl;
            throw;
        }
    }
    
    sqlite3_stmt* stmt = NULL;
    string query = "SELECT * FROM data WHERE id=?";

    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        cerr << "Prepare failed: " << sqlite3_errmsg(db) << endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        throw;
    }

    rc = sqlite3_bind_int(stmt, 1, id);
    if (rc != SQLITE_OK)
    {
        cerr << "bind int failed: " << sqlite3_errmsg(db) << endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        throw;
    }
    
    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW)
    {
        char* name = (char*)sqlite3_column_text(stmt, 1);
        string name_cpp = string(name);

        int nKeypoints = sqlite3_column_int(stmt, 2);
        char* keypointsBinary = (char*)sqlite3_column_blob(stmt, 3);

        vector<KeyPoint> keypoints;
        for (int i = 0; i < nKeypoints; i++)
        {
            vector<char> kp(&keypointsBinary[i*sizeof(KeyPoint)], &keypointsBinary[i*sizeof(KeyPoint)] + sizeof(KeyPoint));
            KeyPoint* kpPtr = reinterpret_cast<KeyPoint*>(&kp[0]);
            keypoints.push_back(*kpPtr);

            kp.clear();
            kp.shrink_to_fit();
        }

        int descriptorSize = sqlite3_column_bytes(stmt, 4);
        char* dPtr = (char*)sqlite3_column_blob(stmt, 4);
        vector<char> dData(dPtr, dPtr + descriptorSize);
        Mat descriptor = DecodeKazeDescriptor(dData, nKeypoints);
        
        sqlite3_finalize(stmt);
        sqlite3_close(db);

        dData.clear();
        dData.shrink_to_fit();

        keypointsBinary = NULL;
        delete[] keypointsBinary;

        return make_tuple(name_cpp, keypoints, descriptor);
    }
    else
    {
        cerr << "No entry with id " << id << " was found." << endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        throw;
    }
}

int WriteEntry(tuple<vector<KeyPoint>, Mat> features, string name)
{
    // First check if user already exists
    /////////////////////////////////////////
    sqlite3 *db = NULL;
    int rc = sqlite3_open_v2("database.db", &db, SQLITE_OPEN_READWRITE, NULL);
    if (rc != SQLITE_OK)
    {
        cerr << "DB open failed: " << sqlite3_errmsg(db) << endl;
        throw;
    }

    sqlite3_stmt* stmt = NULL;
    string query = "SELECT * FROM data";

    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        cerr << "Prepare failed: " << sqlite3_errmsg(db) << endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        throw;
    }
    
    while (true)
    {
        rc = sqlite3_step(stmt);
        if (rc == SQLITE_ROW)
        {
            char* name_c_str = (char*)sqlite3_column_text(stmt, 1);
            string name_str = string(name_c_str);

            if (name_str == name)
            {
                return 1;
                // cerr << " > User already registered." << endl;
            }
        }
        else
            break;
    }

    sqlite3_finalize(stmt);

    /////////////////////////////////////////
    char* descriptorBuffer = EncodeF32Image(get<1>(features));
    /////////////////////////////////////////

    /////////////////////////////////////////
    int keypointCount = get<0>(features).size();
    char* keypointsBuffer = new char[sizeof(KeyPoint)*keypointCount];
    int keypointsByteCounter = 0;
    const void* kpptr = NULL;
    kpptr = keypointsBuffer;

    for (int i = 0; i < keypointCount; i++)
    {
        char* kp = reinterpret_cast<char*>(&get<0>(features)[i]);
        for (int j = 0; j < sizeof(KeyPoint); j++)
        {
            keypointsBuffer[keypointsByteCounter] = kp[j];
            keypointsByteCounter++;
        }    
    }   
    /////////////////////////////////////////

    /////////////////////////////////////////
    stmt = NULL;
    query = "INSERT INTO data(id, name, nKeypoints, keypoints, descriptor) VALUES(NULL, ?, ?, ?, ?)";

    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, NULL);
    if (rc != SQLITE_OK)
    {
        cerr << "Prepare failed: " << sqlite3_errmsg(db) << endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        throw;
    }

    rc = sqlite3_bind_text(stmt, 1, name.c_str(), -1, NULL);
    if (rc != SQLITE_OK)
    {
        cerr << "bind text failed: " << sqlite3_errmsg(db) << endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        throw;
    }

    rc = sqlite3_bind_int(stmt, 2, keypointCount);
    if (rc != SQLITE_OK)
    {
        cerr << "bind int failed: " << sqlite3_errmsg(db) << endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        throw;
    }

    rc = sqlite3_bind_blob(stmt, 3, keypointsBuffer, keypointsByteCounter, SQLITE_STATIC);
    if (rc != SQLITE_OK)
    {
        cerr << "bind 1 failed: " << sqlite3_errmsg(db) << endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        throw;
    }

    rc = sqlite3_bind_blob(stmt, 4, descriptorBuffer, get<1>(features).rows * get<1>(features).cols * sizeof(float), SQLITE_STATIC);
    if (rc != SQLITE_OK)
    {
        cerr << "Prepare failed: " << sqlite3_errmsg(db) << endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        throw;
    }

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE)
    {
        cerr << "Execution failed: " << sqlite3_errmsg(db) << endl;
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        throw;
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    /////////////////////////////////////////

    free(descriptorBuffer);
    delete[] keypointsBuffer;

    return 0;
}

char* EncodeF32Image(Mat& img)
{
    char* buffer = (char*)malloc(sizeof(float) * img.rows * img.cols);
    int ptr = 0;

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            float val = img.at<float>(Point(j, i));
            char* byteVal = reinterpret_cast<char*>(&val);
            
            for (int k = 0; k < sizeof(float); k++, ptr++)
            {
                buffer[ptr] = byteVal[k];
            }          
        }
    }
    
    return buffer;
}

Mat DecodeKazeDescriptor(vector<char> buffer, int nKeypoints)
{
    int cols = 64;
    int rows = nKeypoints;

    Mat img = Mat::zeros(Size(cols, rows), CV_32F);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            vector<char> byteVal(&buffer[(j + i*cols)*sizeof(float)], &buffer[(j + i*cols)*sizeof(float)] + sizeof(float));
            float* val = reinterpret_cast<float*>(&byteVal[0]);

            img.at<float>(Point(j, i)) = *val;
            byteVal.clear();
            byteVal.shrink_to_fit();
        }       
    }

    return img;
}

void PreprocessImage(Mat &src, Mat &dst)
{
    Mat img1, img2, img3, img4, img5, img6, img7, img8;

    // Umbralizar /////////////////////
    threshold(src, img1, 60, 255, THRESH_BINARY);

    Mat framed = Mat::zeros(Size2d(img1.cols + 2, img1.rows + 2), CV_8U);
    Rect r = Rect2d(1, 1, img1.cols, img1.rows);
    img1.copyTo(framed(r));

    Canny(framed, img2, 10, 50);
    
    Mat kernel = getStructuringElement(MORPH_RECT, Size2d(5,5));
    dilate(img2, img2, kernel);

    vector<vector<Point>> contours;
    findContours(img2, contours, RETR_TREE, CHAIN_APPROX_NONE);

    Mat contoursImg = Mat::zeros(img2.size(), CV_8U);
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


    bitwise_and(src, src, img3, contoursImg(r));    
    ///////////////////////////////////

    // CLAHE //////////////////////////
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(6);

    clahe->apply(img3, img3);
    ///////////////////////////////////

    // HFE + HPF //////////////////////
    HPF(img3, img4, FILTER_HFE);
    HPF(img4, img5, FILTER_HPF);
    resize(img5, img6, img3.size());
    bitwise_and(img6, img6, img7, contoursImg(r));
    clahe->apply(img7, img7);
    ///////////////////////////////////   

    // CGF ////////////////////////////
    CGF(img7, img8);
    Mat mask;
    bitwise_not(contoursImg(r), mask);
    normalize(img8, img8, 1.0, 0.0, 4, -1, mask);
    img8.convertTo(img8, CV_8U);
    threshold(img8, img8, 0, 255, THRESH_BINARY);
        
    kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
    dilate(img8, img8, kernel);
    erode(img8, img8, kernel);

    kernel = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
    erode(img8, img8, kernel);
    dilate(img8, img8, kernel);      
    /////////////////////////////////// 

    Mat cropMask = Mat::zeros(img8.size(), CV_8U);
    Rect cropROI = Rect2d(22, 35, img8.cols-60, img8.rows - 35 - 70);
    cropMask(cropROI).setTo(255);
    bitwise_and(img8, cropMask, img8);

    dst = img8.clone();
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

tuple<vector<KeyPoint>, Mat> KAZEDetector(Mat& src)
{
    Mat img = src.clone();

    // SURF ///////////////////////////
    Ptr<KAZE> detector = KAZE::create(false, true);

    vector<KeyPoint> keypoints;
    Mat descriptors;
        
    detector->detectAndCompute(img, noArray(), keypoints, descriptors);
    cout << "KAZE kp: " << to_string(keypoints.size()) << endl;
    return make_tuple(keypoints, descriptors);
    ///////////////////////////////////
}

tuple<Mat, int> FLANNMatcher(tuple<vector<KeyPoint>, Mat> m1, tuple<vector<KeyPoint>, Mat> m2, Mat imgA, Mat imgB)
{
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch(get<1>(m1), get<1>(m2), knn_matches, 2);

    // Filter matches using the Lowe's ratio test.
    const float ratio_thresh = 0.72f;
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
        // Draw matches.
        Mat img_matches;
        drawMatches(imgA.clone(), get<0>(m1), imgB.clone(), get<0>(m2), goodest_matches, img_matches, Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

        // Return mathes image.
        return make_tuple(img_matches.clone(), goodest_matches.size());
    }
    else
    {
        return make_tuple(Mat(), goodest_matches.size());
    }
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

void PerformTest(Mat& src, Mat& dst)
{
    Mat img, img2, img3, img4, img5, img6, img7, img8, img9, img10;

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
    HPF(img5, img6, FILTER_HFE);
    resize(img6, img8, img5.size());
    bitwise_and(img8, img8, imga, contoursImg(r));
    clahe->apply(imga, imga);
    imshow("HFE", imga);
    ///////////////////////////////////   

    // HPF //////////////////////
    HPF(img5, img6, FILTER_HPF);
    resize(img6, img8, img5.size());
    bitwise_and(img8, img8, imgb, contoursImg(r));
    clahe->apply(imgb, imgb);
    imshow("HPF", imgb);
    ///////////////////////////////////   

    // HPF + HFE //////////////////////
    HPF(img5, img6, FILTER_HPF);
    HPF(img6, img7, FILTER_HFE);        
    resize(img7, img8, img5.size());
    bitwise_and(img8, img8, imgc, contoursImg(r));
    clahe->apply(imgc, imgc);
    imshow("HPF + HFE", imgc);
    ///////////////////////////////////   

    // HFE + HPF //////////////////////
    HPF(img5, img6, FILTER_HFE);
    HPF(img6, img7, FILTER_HPF);
    resize(img7, img8, img5.size());
    bitwise_and(img8, img8, imgd, contoursImg(r));
    clahe->apply(imgd, imgd);
    imshow("HFE + HPF", imgd);
    ///////////////////////////////////   

    // CGF ////////////////////////////
    CGF(imgd, imge);
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

    Mat surfmask = Mat::zeros(imge.size(), CV_8U);
    Rect surfROI = Rect2d(22, 35, imge.cols-70, imge.rows - 35 - 70);
    surfmask(surfROI).setTo(255);
    bitwise_and(imge, surfmask, imge);

    // SURF ///////////////////////////
    Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(60000);

    vector<KeyPoint> keypoints;
    Mat descriptors;
        
    detector->detect(imge(surfROI), keypoints);
    // detector->detectAndCompute(imge, surfmask, keypoints, descriptors);
    // std::cout << keypoints.size() << std::endl;

    Mat imgf;
    drawKeypoints(imge(surfROI), keypoints, imgf, Scalar(255, 100, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("SURF", imgf);
    ///////////////////////////////////

    // SIFT ///////////////////////////
    Ptr<SIFT> detector2 = SIFT::create(40);

    vector<KeyPoint> keypoints2;
        
    detector2->detect(imge(surfROI), keypoints2);
    // detector->detectAndCompute(imge, surfmask, keypoints, descriptors);
    // std::cout << keypoints.size() << std::endl;

    Mat imgg;
    drawKeypoints(imge(surfROI), keypoints2, imgg, Scalar(255, 100, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("SIFT", imgg);
    ///////////////////////////////////

    // ORB ////////////////////////////
    Ptr<ORB> detector3 = ORB::create(40);

    vector<KeyPoint> keypoints3;
    
    detector3->detect(imge(surfROI), keypoints3);
    // detector->detectAndCompute(imge, surfmask, keypoints, descriptors);
    // std::cout << keypoints.size() << std::endl;

    Mat imgh;
    drawKeypoints(imge(surfROI), keypoints3, imgh, Scalar(255, 100, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("ORB", imgh);
    ///////////////////////////////////
}
 
void CGF(Mat &src, Mat &dst)
{
    float sigma = 5; //5-pixel width
    float DeltaF = 1.12; //[0.5, 2,5]

    sigma = sig;
    DeltaF = dF/100.0f;

    float fc = ((1/M_PI)*sqrt(log(2)/2)*((pow(2, DeltaF) + 1)/(pow(2, DeltaF) - 1)))/sigma;

    Mat g = Mat::zeros(Size2d(30, 30), CV_32F);
    Mat G = Mat::zeros(Size2d(30, 30), CV_32F);
    Mat Gim = Mat::zeros(Size2d(30, 30), CV_32F);
    for (int i = 0; i < g.cols; i++)
    {
        for (int j = 0; j < g.rows; j++)
        {
            g.at<float>(Point(i,j)) = (1/(2*M_PI*pow(sigma, 2))) * exp(-(pow(i - g.cols / 2, 2) + pow(j - g.rows / 2, 2))/(2*pow(sigma, 2)));
            G.at<float>(Point(i,j)) = g.at<float>(Point(i,j)) * cos(2*M_PI*fc*sqrt(pow(i - g.cols / 2, 2) + pow(j - g.rows / 2, 2)));
            Gim.at<float>(Point(i,j)) = g.at<float>(Point(i,j)) * sin(2*M_PI*fc*sqrt(pow(i - g.cols / 2, 2) + pow(j - g.rows / 2, 2)));
        }        
    }

    Mat res;
    filter2D(src, res, CV_32F, G);
    dst = res.clone();
}

Mat DFTModule(Mat src[], bool shift)
{
    Mat magImg;
    magnitude(src[0], src[1], magImg); // Magnitud = planes[0]

    magImg += Scalar::all(1); // Cambia a escala log
    log(magImg, magImg); 

    // Recorta el espectro si tiene rows o cols impares
    magImg = magImg(Rect(0, 0, magImg.cols & -2, magImg.rows & -2));

    if(shift)
        FFTShift(magImg, magImg);

    normalize(magImg, magImg, 0, 1, NORM_MINMAX);

    return magImg;
}

void HPF(Mat& src, Mat& dst, uint8_t filterType)
{
    Mat padded; // Expande la imagen al tamaño óptimo
    int m = getOptimalDFTSize(src.rows);
    int n = getOptimalDFTSize(src.cols); // Añade valores cero en el borde
    copyMakeBorder(src, padded, 0, m - src.rows, 0, n - src.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexImg;
    merge(planes, 2, complexImg); // Añade un plano complejo

    /// DFT
    dft(complexImg, complexImg);
    split(complexImg, planes);
    ///

    /// FILTER
    Mat H = Mat::zeros(src.size(), CV_32F);
    Mat filt = Mat::zeros(src.size(), CV_32F);
    Mat HF = complexImg.clone();

    // float D0 = 40.0f;
    // float k1 = filterType ? 0.5f : -4;
    // float k2 = filterType ? 0.75f : 10.9;
    float D0;
    float k1, k2;

    if(filterType)
    {
        D0 = (d0hfe > 0) ? d0hfe : 1;
        k1 = (k1hfe-5000) / 100.0f;
        k2 = (k2hfe-5000) / 100.0f;
    }
    else
    {
        D0 = (d0hpf > 0) ? d0hpf : 1;
        k1 = (k1hpf-5000) / 1000.0f;
        k2 = (k2hpf-5000) / 1000.0f;
    }

    // float a = 10.9, b = -4;
    
    for (int i = 0; i < H.cols; i++)
    {
        for (int j = 0; j < H.rows; j++)
        {
            H.at<float>(Point(i,j)) = 1.0 - exp( -(pow(i - H.cols / 2, 2) + pow(j - H.rows / 2, 2)) / (2 * pow(D0, 2)) );
        }
        
    }

    filt = k1 + k2*H;

    FFTShift(HF, HF);
    split(HF, planes);
    for (int i = 0; i < filt.cols; i++)
    {
        for (int j = 0; j < filt.rows; j++)
        {
            planes[0].at<float>(Point(i,j)) *= filt.at<float>(Point(i,j));
            planes[1].at<float>(Point(i,j)) *= filt.at<float>(Point(i,j));
        }
    } 

    Mat filteredImg;
    merge(planes, 2, filteredImg);

    FFTShift(filteredImg, filteredImg);
    ///

    /// IDFT
    Mat result;
    dft(filteredImg, result, DFT_INVERSE | DFT_REAL_OUTPUT);
    normalize(result, result, 0, 1, NORM_MINMAX);
    result.convertTo(result, CV_8U, 255);
    equalizeHist(result, result);
    ///

    dst = result.clone();
}

void FFTShift(const Mat& src, Mat &dst)
{
    dst = src.clone();
    int cx = dst.cols / 2;
    int cy = dst.rows / 2;

    Mat q1(dst, Rect(0, 0, cx, cy));
    Mat q2(dst, Rect(cx, 0, cx, cy));
    Mat q3(dst, Rect(0, cy, cx, cy));
    Mat q4(dst, Rect(cx, cy, cx, cy));

    Mat temp;
    q1.copyTo(temp);
    q4.copyTo(q1);
    temp.copyTo(q4);
    q2.copyTo(temp);
    q3.copyTo(q2);
    temp.copyTo(q3);
}