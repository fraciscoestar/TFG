#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <wiringPi.h>

#define FILTER_HPF 0
#define FILTER_HFE 1

using namespace std;
using namespace cv;

void HPF(Mat& src, Mat& dst, uint8_t filterType);
void FFTShift(const Mat& src, Mat &dst);
void test();
Mat DFTModule(Mat src[], bool shift);
void CGF(Mat &src, Mat &dst, float F, float sigma, float theta);

int main(int argc, char const *argv[])
{
    Mat img, img2, img3, img4, img5, img6, img7, img8, img9;
    int gamma_ = 10;
    int F = 10, sigma = 32, theta = 0, lambda = 115;

    namedWindow("Gamma bar");
    createTrackbar("Gamma", "Gamma bar", &gamma_, 1000);
    createTrackbar("F", "Gamma bar", &F, 100);
    createTrackbar("Sigma", "Gamma bar", &sigma, 100);
    createTrackbar("Theta", "Gamma bar", &theta, 1000);
    createTrackbar("Lambda", "Gamma bar", &lambda, 1000);
    
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

    imshow("Contour", contoursImg);

    bitwise_and(img2, img2, img5, contoursImg(r));

    imshow("Umbralized", img5);
    
    ///////////////////////////////////

    // CLAHE //////////////////////////
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);

    clahe->apply(img5, img5);
    ///////////////////////////////////

    // HPF ////////////////////////////
    HPF(img5, img6, FILTER_HPF);
    resize(img6, img7, img5.size());
    bitwise_and(img7, img7, img8, contoursImg(r));
    clahe->apply(img8, img8);
    ///////////////////////////////////

    imshow("CLAHE", img5);
    imshow("HPF", img8);

    

    while(1)
    {
        //CGF(img3, img, F/10.0, sigma/10.0, 0);
        Mat kernel = getGaborKernel(Size2d(21, 21), sigma/10.0, theta/100.0, lambda/100.0, gamma_/10.0);
        filter2D(img8, img9, CV_8U, kernel);

        imshow("Gabor filter", img9);
        imshow("Kernel", kernel);
        waitKey(1);
    }
    return 0;
}
 
void CGF(Mat &src, Mat &dst, float F, float sigma, float theta)
{
    float alpha = 1.0 / (2*M_PI*sigma);
    float Fuv;
    Mat g = Mat::zeros(src.size(), CV_32F);
    Mat G[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };

    for (int i = 0; i < src.cols; i++)
    {
        for (int j = 0; j < src.rows; j++)
        {      
            Fuv = (sqrt(2*M_PI) / 2.0) * alpha * exp(-pow(sqrt(pow(i, 2) + pow(j, 2)) - F, 2) / (2*pow(alpha, 2)));
            g.at<float>(Point(i, j)) = (1.0 / (2*M_PI*pow(sigma, 2))) * exp(-(pow(i, 2) + pow(j, 2)) / (2*pow(sigma, 2)));
            const std::complex<float> temp(0, 2*M_PI*Fuv*sqrt(pow(i, 2) + pow(j, 2)));           
            std::complex<float> Gxy(0, 0);

            Gxy = g.at<float>(Point(i, j)) * exp(temp);
            G[0].at<float>(Point(i, j)) = Gxy.real();
            G[1].at<float>(Point(i, j)) = Gxy.imag();
        }      
    }

    normalize(G[0], G[0], 0, 1, NORM_MINMAX);
    G[0].convertTo(G[0], CV_8U, 255);
    normalize(G[1], G[1], 0, 1, NORM_MINMAX);
    G[1].convertTo(G[1], CV_8U, 255);
    
    imshow("CGF re", G[0]);
    imshow("CGF im", G[1]);
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

    float D0 = 40.0f;
    float k1 = filterType ? 0.5f : -4;
    float k2 = filterType ? 0.75f : 10.9;

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
