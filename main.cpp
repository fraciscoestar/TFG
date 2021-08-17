#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <wiringPi.h>

#define FILTER_HPF 0
#define FILTER_HFE 1

using namespace std;
using namespace cv;

void HPF(Mat& src, Mat& dst, uint8_t filterType);
void FFTShift(const Mat& src, Mat &dst);
void test();
Mat DFTModule(Mat src[], bool shift);
void CGF(Mat &src, Mat &dst);

int main(int argc, char const *argv[])
{
    Mat img, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11;
    
    // Activate Lights ////////////////
    wiringPiSetup();
    pinMode(4, OUTPUT);
    pinMode(5, OUTPUT);

    digitalWrite(4, HIGH); //980nm
    digitalWrite(5, HIGH); //850nm
    ///////////////////////////////////

    // Take image /////////////////////
    // img = imread("Lenna.png", IMREAD_GRAYSCALE);
    VideoCapture vc(0);
    vc.read(img);
    cvtColor(img, img, COLOR_BGR2GRAY);
    Rect fingerRegion = Rect(255, 0, 465-255, img.rows-40);
    img2 = img(fingerRegion);
    imshow("Img", img2);
    ///////////////////////////////////

    // Deactivate Lights //////////////
    digitalWrite(4, LOW);
    digitalWrite(5, LOW);
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
    ///////////////////////////////////

    // HFE + HPF //////////////////////
    HPF(img5, img6, FILTER_HFE);
    HPF(img6, img7, FILTER_HPF);
    resize(img7, img8, img5.size());
    bitwise_and(img8, img8, img9, contoursImg(r));
    clahe->apply(img9, img9);    
    ///////////////////////////////////

    // CGF ////////////////////////////
    CGF(img9, img10);
    Mat mask;
    bitwise_not(contoursImg(r), mask);
    normalize(img10, img10, 1.0, 0.0, 4, -1, mask); 
    img10.convertTo(img10, CV_8U);
    threshold(img10, img10, 0, 255, THRESH_BINARY);

    kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
    dilate(img10, img10, kernel);
    erode(img10, img10, kernel);

    kernel = getStructuringElement(MORPH_ELLIPSE, Size(3,3));
    erode(img10, img10, kernel);
    dilate(img10, img10, kernel);
    ///////////////////////////////////

    // SURF ///////////////////////////
    Ptr<xfeatures2d::SURF> detector = xfeatures2d::SURF::create(60000);
    vector<KeyPoint> keypoints;
    Mat surfMask = Mat::zeros(img10.size(), CV_8U);
    Rect surfROI = Rect2d(23, 35, img10.cols-75, img10.rows -35 -70);
    surfMask(surfROI).setTo(255);
    bitwise_and(img10, surfMask, img10);

    detector->detect(img10(surfROI), keypoints);
    
    drawKeypoints(img10(surfROI), keypoints, img11, Scalar(255, 100, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    ///////////////////////////////////

    imshow("CLAHE", img5);
    imshow("HFE+HPF", img9);
    imshow("CGF", img10);
    imshow("keypoints", img11);

    waitKey(0);
    return 0;
}
 
void CGF(Mat &src, Mat &dst)
{
    float sigma = 5;
    float deltaF = 4.36;

    float fc = ((1/M_PI)*sqrt(log(2)/2)*((pow(2, deltaF) + 1)/(pow(2, deltaF) - 1)))/sigma;
    Mat g = Mat::zeros(Size2d(30,30), CV_32F);
    Mat G = Mat::zeros(Size2d(30,30), CV_32F);
    for (int i = 0; i < g.cols; i++)
    {
        for (int j = 0; j < g.rows; j++)
        {
            g.at<float>(Point(i,j)) = (1/(2*M_PI*pow(sigma, 2))) * exp(-(pow(i - g.cols / 2, 2) + pow(j - g.rows / 2, 2))/(2*pow(sigma, 2)));
            G.at<float>(Point(i,j)) = g.at<float>(Point(i,j)) * cos(2*M_PI*fc*sqrt(pow(i - g.cols / 2, 2) + pow(j - g.rows / 2, 2)));
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

    float D0, k1, k2;

    if(filterType)
    {
        D0 = 10;
        k1 = 3.0;
        k2 = 23.27;
    }
    else
    {
        D0 = 40;
        k1 = -19.50;
        k2 = 4.63;
    }
    
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

void test()
{
    Mat img = imread("Lenna.png", IMREAD_GRAYSCALE);
    if(img.empty())
        exit(1);

    Mat padded; // Expande la imagen al tamaño óptimo
    int m = getOptimalDFTSize(img.rows);
    int n = getOptimalDFTSize(img.cols); // Añade valores cero en el borde
    copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexImg;
    merge(planes, 2, complexImg); // Añade un plano complejo

    /// DFT
    dft(complexImg, complexImg);
    split(complexImg, planes);

    /*magnitude(planes[0], planes[1], planes[0]); // Magnitud = planes[0]
    Mat magImg = planes[0];

    magImg += Scalar::all(1); // Cambia a escala log
    log(magImg, magImg); 

    // Recorta el espectro si tiene rows o cols impares
    magImg = magImg(Rect(0, 0, magImg.cols & -2, magImg.rows & -2));

    //FFT SHift
    FFTShift(magImg, magImg);

    normalize(magImg, magImg, 0, 1, NORM_MINMAX);
    //normalize(phasevals, phasevals, 0, 1, NORM_MINMAX);*/
    Mat magImg = DFTModule(planes, true);

    imshow("Input", img);
    imshow("Magnitud", magImg);
    ///

    /// FILTER
    Mat H = Mat::zeros(magImg.size(), CV_32F);
    Mat filt = Mat::zeros(magImg.size(), CV_32F);
    Mat HFE = complexImg.clone();

    float D0 = 40.0f, k1 = 0.5f, k2 = 0.75f;
    
    for (int i = 0; i < H.cols; i++)
    {
        for (int j = 0; j < H.rows; j++)
        {
            H.at<float>(Point(i,j)) = 1.0 - exp( -(pow(i - H.cols / 2, 2) + pow(j - H.rows / 2, 2)) / (2 * pow(D0, 2)) );
        }
        
    }

    filt = k1 + k2*H;
    imshow("filt", filt);

    FFTShift(HFE, HFE);
    split(HFE, planes);
    for (int i = 0; i < H.cols; i++)
    {
        for (int j = 0; j < H.rows; j++)
        {
            planes[0].at<float>(Point(i,j)) *= filt.at<float>(Point(i,j));
            planes[1].at<float>(Point(i,j)) *= filt.at<float>(Point(i,j));
        }
    } 

    Mat filteredImg;
    merge(planes, 2, filteredImg);

    FFTShift(filteredImg, filteredImg);
    imshow("esta", DFTModule(planes, false));

    ///

    // /// IDFT
    // Mat inverseTransform;
    // dft(complexImg, inverseTransform, DFT_INVERSE | DFT_REAL_OUTPUT);
    // normalize(inverseTransform, inverseTransform, 0, 1, NORM_MINMAX);
    // imshow("Reconstruida", inverseTransform);
    // ///

    /// IDFT 2
    Mat inverseTransform2, result;
    dft(filteredImg, inverseTransform2, DFT_INVERSE | DFT_REAL_OUTPUT);
    normalize(inverseTransform2, inverseTransform2, 0, 1, NORM_MINMAX);
    inverseTransform2.convertTo(inverseTransform2, CV_8U, 255);
    equalizeHist(inverseTransform2, inverseTransform2);
    imshow("Resultado", inverseTransform2);
    ///

    split(complexImg, planes);
    imshow("CI re", planes[0]);
    imshow("CI im", planes[1]);

    split(filteredImg, planes);
    imshow("HFE re", planes[0]);
    imshow("HFE im", planes[1]);
}
