#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#define FILTER_HPF 0
#define FILTER_HFE 1

using namespace std;
using namespace cv;

void HPF(Mat& src, Mat& dst, uint8_t filterType);
void FFTShift(const Mat& src, Mat &dst);
void test();
Mat DFTModule(Mat src[], bool shift);
void CGF(Mat &src, Mat &dst);

int d0hfe = 10, d0hpf = 40, k1hfe = 5300, k2hfe = 7327, k1hpf = 3050, k2hpf = 5463, sig = 5, dF=436;

int main(int argc, char const *argv[])
{
    Mat img, img2, img3, img4, img5, img6, img7, img8, img9, img10;

    namedWindow("Settings");
    createTrackbar("D0 HFE", "Settings", &d0hfe, 1000);
    createTrackbar("k1 HFE", "Settings", &k1hfe, 10000);
    createTrackbar("k2 HFE", "Settings", &k2hfe, 10000);
    createTrackbar("D0 HPF", "Settings", &d0hpf, 1000);
    createTrackbar("k1 HPF", "Settings", &k1hpf, 10000);
    createTrackbar("k2 HPF", "Settings", &k2hpf, 10000);
    createTrackbar("Sigma", "Settings", &sig, 100);
    createTrackbar("DeltaF", "Settings", &dF, 1000);

    // Take image /////////////////////
    img = imread("original.png");
    cvtColor(img, img, COLOR_BGR2GRAY);
    img2 = img.clone();
    imshow("Img", img2);
    ///////////////////////////////////

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

    while (1)
    {
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

        waitKey(1);
    }

    waitKey(0);
    return 0;
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