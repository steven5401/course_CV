#include <iostream>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <random>
#include <algorithm>
#include <cmath>
using namespace std;
using namespace cv;
//#define DEBUG
#ifdef DEBUG
    #define D(x) x
#else
    #define D(x)
#endif

Mat LaplaceMask1(const Mat src, int thresh) {
    Mat kernel1(3, 3, CV_16S, Scalar(0));
    kernel1.at<short>(0,1) = 1;
    kernel1.at<short>(1,0) = 1;
    kernel1.at<short>(1,1) = -4;
    kernel1.at<short>(1,2) = 1;
    kernel1.at<short>(2,1) = 1;
    Mat ret;
    filter2D(src, ret, CV_16S, kernel1, Point(-1,-1), 0, BORDER_DEFAULT);
    convertScaleAbs(ret, ret);
    threshold(ret, ret, thresh, 255, THRESH_BINARY);
    return ret;
}

Mat LaplaceMask2(const Mat src, int thresh) {
    Mat kernel1(3, 3, CV_16S, Scalar(0));
    kernel1.at<short>(0,0) = 1;
    kernel1.at<short>(0,1) = 1;
    kernel1.at<short>(0,2) = 1;
    kernel1.at<short>(1,0) = 1;
    kernel1.at<short>(1,1) = -8;
    kernel1.at<short>(1,2) = 1;
    kernel1.at<short>(2,0) = 1;
    kernel1.at<short>(2,1) = 1;
    kernel1.at<short>(2,2) = 1;
    Mat ret;
    filter2D(src, ret, CV_16S, kernel1, Point(-1,-1), 0, BORDER_DEFAULT);
    convertScaleAbs(ret, ret, 1.0/3.0);
    threshold(ret, ret, thresh, 255, THRESH_BINARY);
    return ret;
}

Mat MinimumVarianceLaplacian(const Mat src, int thresh) {
    Mat kernel1(3, 3, CV_16S, Scalar(0));
    kernel1.at<short>(0,0) = 2;
    kernel1.at<short>(0,1) = -1;
    kernel1.at<short>(0,2) = 2;
    kernel1.at<short>(1,0) = -1;
    kernel1.at<short>(1,1) = -4;
    kernel1.at<short>(1,2) = -1;
    kernel1.at<short>(2,0) = 2;
    kernel1.at<short>(2,1) = -1;
    kernel1.at<short>(2,2) = 2;
    Mat ret;
    filter2D(src, ret, CV_16S, kernel1, Point(-1,-1), 0, BORDER_DEFAULT);
    convertScaleAbs(ret, ret, 1.0/3.0);
    threshold(ret, ret, thresh, 255, THRESH_BINARY);
    return ret;
}

Mat LoG(const Mat src, int thresh) {
    Mat kernel1(11, 11, CV_16S, Scalar(0));
    kernel1.at<short>(0,3) = -1;
    kernel1.at<short>(0,4) = -1;
    kernel1.at<short>(0,5) = -2;
    kernel1.at<short>(0,6) = -1;
    kernel1.at<short>(0,7) = -1;
    kernel1.at<short>(1,2) = -2;
    kernel1.at<short>(1,3) = -4;
    kernel1.at<short>(1,4) = -8;
    kernel1.at<short>(1,5) = -9;
    kernel1.at<short>(1,6) = -8;
    kernel1.at<short>(1,7) = -4;
    kernel1.at<short>(1,8) = -2;
    kernel1.at<short>(2,1) = -2;
    kernel1.at<short>(2,2) = -7;
    kernel1.at<short>(2,3) = -15;
    kernel1.at<short>(2,4) = -22;
    kernel1.at<short>(2,5) = -23;
    kernel1.at<short>(2,6) = -22;
    kernel1.at<short>(2,7) = -15;
    kernel1.at<short>(2,8) = -7;
    kernel1.at<short>(2,9) = -2;
    kernel1.at<short>(3,0) = -1;
    kernel1.at<short>(3,1) = -4;
    kernel1.at<short>(3,2) = -15;
    kernel1.at<short>(3,3) = -24;
    kernel1.at<short>(3,4) = -14;
    kernel1.at<short>(3,5) = -1;
    kernel1.at<short>(3,6) = -14;
    kernel1.at<short>(3,7) = -24;
    kernel1.at<short>(3,8) = -15;
    kernel1.at<short>(3,9) = -4;
    kernel1.at<short>(3,10) = -1;
    kernel1.at<short>(4,0) = -1;
    kernel1.at<short>(4,1) = -8;
    kernel1.at<short>(4,2) = -22;
    kernel1.at<short>(4,3) = -14;
    kernel1.at<short>(4,4) = 52;
    kernel1.at<short>(4,5) = 103;
    kernel1.at<short>(4,6) = 52;
    kernel1.at<short>(4,7) = -14;
    kernel1.at<short>(4,8) = -22;
    kernel1.at<short>(4,9) = -8;
    kernel1.at<short>(4,10) = -1;
    kernel1.at<short>(5,0) = -2;
    kernel1.at<short>(5,1) = -9;
    kernel1.at<short>(5,2) = -23;
    kernel1.at<short>(5,3) = -1;
    kernel1.at<short>(5,4) = 103;
    kernel1.at<short>(5,5) = 178;
    kernel1.at<short>(5,6) = 103;
    kernel1.at<short>(5,7) = -1;
    kernel1.at<short>(5,8) = -23;
    kernel1.at<short>(5,9) = -9;
    kernel1.at<short>(5,10) = -2;//
    kernel1.at<short>(10,3) = -1;
    kernel1.at<short>(10,4) = -1;
    kernel1.at<short>(10,5) = -2;
    kernel1.at<short>(10,6) = -1;
    kernel1.at<short>(10,7) = -1;
    kernel1.at<short>(9,2) = -2;
    kernel1.at<short>(9,3) = -4;
    kernel1.at<short>(9,4) = -8;
    kernel1.at<short>(9,5) = -9;
    kernel1.at<short>(9,6) = -8;
    kernel1.at<short>(9,7) = -4;
    kernel1.at<short>(9,8) = -2;
    kernel1.at<short>(8,1) = -2;
    kernel1.at<short>(8,2) = -7;
    kernel1.at<short>(8,3) = -15;
    kernel1.at<short>(8,4) = -22;
    kernel1.at<short>(8,5) = -23;
    kernel1.at<short>(8,6) = -22;
    kernel1.at<short>(8,7) = -15;
    kernel1.at<short>(8,8) = -7;
    kernel1.at<short>(8,9) = -2;
    kernel1.at<short>(7,0) = -1;
    kernel1.at<short>(7,1) = -4;
    kernel1.at<short>(7,2) = -15;
    kernel1.at<short>(7,3) = -24;
    kernel1.at<short>(7,4) = -14;
    kernel1.at<short>(7,5) = -1;
    kernel1.at<short>(7,6) = -14;
    kernel1.at<short>(7,7) = -24;
    kernel1.at<short>(7,8) = -15;
    kernel1.at<short>(7,9) = -4;
    kernel1.at<short>(7,10) = -1;
    kernel1.at<short>(6,0) = -1;
    kernel1.at<short>(6,1) = -8;
    kernel1.at<short>(6,2) = -22;
    kernel1.at<short>(6,3) = -14;
    kernel1.at<short>(6,4) = 52;
    kernel1.at<short>(6,5) = 103;
    kernel1.at<short>(6,6) = 52;
    kernel1.at<short>(6,7) = -14;
    kernel1.at<short>(6,8) = -22;
    kernel1.at<short>(6,9) = -8;
    kernel1.at<short>(6,10) = -1;
    Mat ret;
    filter2D(src, ret, CV_16S, kernel1, Point(-1,-1), 0, BORDER_DEFAULT);
    threshold(ret, ret, thresh, 255, THRESH_BINARY);
    convertScaleAbs(ret, ret);
    return ret;
}

Mat DoG(const Mat src, int thresh) {
    Mat kernel1(11, 11, CV_16S, Scalar(0));
    kernel1.at<short>(0,0) = -1;
    kernel1.at<short>(0,1) = -3;
    kernel1.at<short>(0,2) = -4;
    kernel1.at<short>(0,3) = -6;
    kernel1.at<short>(0,4) = -7;
    kernel1.at<short>(0,5) = -8;
    kernel1.at<short>(0,6) = -7;
    kernel1.at<short>(0,7) = -6;
    kernel1.at<short>(0,8) = -4;
    kernel1.at<short>(0,9) = -3;
    kernel1.at<short>(0,10) = -1;
    kernel1.at<short>(1,0) = -3;
    kernel1.at<short>(1,1) = -5;
    kernel1.at<short>(1,2) = -8;
    kernel1.at<short>(1,3) = -11;
    kernel1.at<short>(1,4) = -13;
    kernel1.at<short>(1,5) = -13;
    kernel1.at<short>(1,6) = -13;
    kernel1.at<short>(1,7) = -11;
    kernel1.at<short>(1,8) = -8;
    kernel1.at<short>(1,9) = -5;
    kernel1.at<short>(1,10) = -3;
    kernel1.at<short>(2,0) = -4;
    kernel1.at<short>(2,1) = -8;
    kernel1.at<short>(2,2) = -12;
    kernel1.at<short>(2,3) = -16;
    kernel1.at<short>(2,4) = -17;
    kernel1.at<short>(2,5) = -17;
    kernel1.at<short>(2,6) = -17;
    kernel1.at<short>(2,7) = -16;
    kernel1.at<short>(2,8) = -12;
    kernel1.at<short>(2,9) = -8;
    kernel1.at<short>(2,10) = -4;
    kernel1.at<short>(3,0) = -6;
    kernel1.at<short>(3,1) = -11;
    kernel1.at<short>(3,2) = -16;
    kernel1.at<short>(3,3) = -16;
    kernel1.at<short>(3,4) = 0;
    kernel1.at<short>(3,5) = 15;
    kernel1.at<short>(3,6) = 0;
    kernel1.at<short>(3,7) = -16;
    kernel1.at<short>(3,8) = -16;
    kernel1.at<short>(3,9) = -11;
    kernel1.at<short>(3,10) = -6;
    kernel1.at<short>(4,0) = -7;
    kernel1.at<short>(4,1) = -13;
    kernel1.at<short>(4,2) = -17;
    kernel1.at<short>(4,3) = 0;
    kernel1.at<short>(4,4) = 85;
    kernel1.at<short>(4,5) = 160;
    kernel1.at<short>(4,6) = 85;
    kernel1.at<short>(4,7) = 0;
    kernel1.at<short>(4,8) = -17;
    kernel1.at<short>(4,9) = -13;
    kernel1.at<short>(4,10) = -7;
    kernel1.at<short>(5,0) = -8;
    kernel1.at<short>(5,1) = -13;
    kernel1.at<short>(5,2) = -17;
    kernel1.at<short>(5,3) = 15;
    kernel1.at<short>(5,4) = 160;
    kernel1.at<short>(5,5) = 283;
    kernel1.at<short>(5,6) = 160;
    kernel1.at<short>(5,7) = 15;
    kernel1.at<short>(5,8) = -17;
    kernel1.at<short>(5,9) = -13;
    kernel1.at<short>(5,10) = -8;
    kernel1.at<short>(10,0) = -1;
    kernel1.at<short>(10,1) = -3;
    kernel1.at<short>(10,2) = -4;
    kernel1.at<short>(10,3) = -6;
    kernel1.at<short>(10,4) = -7;
    kernel1.at<short>(10,5) = -8;
    kernel1.at<short>(10,6) = -7;
    kernel1.at<short>(10,7) = -6;
    kernel1.at<short>(10,8) = -4;
    kernel1.at<short>(10,9) = -3;
    kernel1.at<short>(10,10) = -1;
    kernel1.at<short>(9,0) = -3;
    kernel1.at<short>(9,1) = -5;
    kernel1.at<short>(9,2) = -8;
    kernel1.at<short>(9,3) = -11;
    kernel1.at<short>(9,4) = -13;
    kernel1.at<short>(9,5) = -13;
    kernel1.at<short>(9,6) = -13;
    kernel1.at<short>(9,7) = -11;
    kernel1.at<short>(9,8) = -8;
    kernel1.at<short>(9,9) = -5;
    kernel1.at<short>(9,10) = -3;
    kernel1.at<short>(8,0) = -4;
    kernel1.at<short>(8,1) = -8;
    kernel1.at<short>(8,2) = -12;
    kernel1.at<short>(8,3) = -16;
    kernel1.at<short>(8,4) = -17;
    kernel1.at<short>(8,5) = -17;
    kernel1.at<short>(8,6) = -17;
    kernel1.at<short>(8,7) = -16;
    kernel1.at<short>(8,8) = -12;
    kernel1.at<short>(8,9) = -8;
    kernel1.at<short>(8,10) = -4;
    kernel1.at<short>(7,0) = -6;
    kernel1.at<short>(7,1) = -11;
    kernel1.at<short>(7,2) = -16;
    kernel1.at<short>(7,3) = -16;
    kernel1.at<short>(7,4) = 0;
    kernel1.at<short>(7,5) = 15;
    kernel1.at<short>(7,6) = 0;
    kernel1.at<short>(7,7) = -16;
    kernel1.at<short>(7,8) = -16;
    kernel1.at<short>(7,9) = -11;
    kernel1.at<short>(7,10) = -6;
    kernel1.at<short>(6,0) = -7;
    kernel1.at<short>(6,1) = -13;
    kernel1.at<short>(6,2) = -17;
    kernel1.at<short>(6,3) = 0;
    kernel1.at<short>(6,4) = 85;
    kernel1.at<short>(6,5) = 160;
    kernel1.at<short>(6,6) = 85;
    kernel1.at<short>(6,7) = 0;
    kernel1.at<short>(6,8) = -17;
    kernel1.at<short>(6,9) = -13;
    kernel1.at<short>(6,10) = -7;
    Mat ret;
    filter2D(src, ret, CV_16S, kernel1, Point(-1,-1), 0, BORDER_DEFAULT);
    threshold(ret, ret, thresh, 255, THRESH_BINARY);
    convertScaleAbs(ret, ret);
    return ret;
}

int main(int argc, char** argv) {
    Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    CV_Assert(img_gray.depth() == CV_8U);
    int channels = img_gray.channels();
    int nRows = img_gray.rows;
    int nCols = img_gray.cols * channels;
    imwrite("laplaceMask1.bmp", LaplaceMask1(img_gray, 32));
    imwrite("laplaceMask2.bmp", LaplaceMask2(img_gray, 24));
    imwrite("minimumVarianceLaplacian.bmp", MinimumVarianceLaplacian(img_gray, 24));
    imwrite("LoG.bmp", LoG(img_gray, 3000));
    imwrite("DoG.bmp", DoG(img_gray, 10000));
}