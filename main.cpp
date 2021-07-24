#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <wiringPi.h>

using namespace std;
using namespace cv;

void HFE(Mat& src, Mat& dst);
void FFTShift(const Mat& src, Mat &dst);
void test();
Mat DFTModule(Mat src[], bool shift);

int main(int argc, char const *argv[])
{
    Mat img, img2, img3;
    
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
    Rect fingerRegion = Rect(255, 0, 465-255, img.rows);
    img2 = img(fingerRegion);
    imshow("Img", img2);
    ///////////////////////////////////

    // CLAHE //////////////////////////
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);

    clahe->apply(img2, img2);
    ///////////////////////////////////

    // HFE ////////////////////////////
    HFE(img2, img3);
    ///////////////////////////////////

    // Deactivate Lights ////////////////
    digitalWrite(4, LOW);
    digitalWrite(5, LOW);
    ///////////////////////////////////

    imshow("CLAHE", img2);
    imshow("HFE", img3);

    waitKey(0);

    return 0;
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

void HFE(Mat& src, Mat& dst)
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
    Mat H2 = Mat::zeros(src.size(), CV_32F);
    Mat filt = Mat::zeros(src.size(), CV_32F);
    Mat HFE = complexImg.clone();

    float D0 = 40.0f, k1 = 0.5f, k2 = 0.75f;
    float a = 10.9, b = -4;
    
    for (int i = 0; i < H.cols; i++)
    {
        for (int j = 0; j < H.rows; j++)
        {
            H.at<float>(Point(i,j)) = 1.0 - exp( -(pow(i - H.cols / 2, 2) + pow(j - H.rows / 2, 2)) / (2 * pow(D0, 2)) );
            H2.at<float>(Point(i,j)) = a*(1.0 - exp( -(pow(i - H.cols / 2, 2) + pow(j - H.rows / 2, 2)) / (2 * pow(D0, 2)))) + b;
        }
        
    }

    //filt = k1 + k2*H;
    filt = H2;

    FFTShift(HFE, HFE);
    split(HFE, planes);
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
