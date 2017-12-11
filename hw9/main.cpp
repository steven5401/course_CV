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
Mat myConv(const Mat src, const Mat kernel, int origin_r, int origin_c) {
    Mat big(src.rows + kernel.rows - 1, src.cols + kernel.cols - 1, CV_8UC1, Scalar(0));
    Mat roi(big, Rect(origin_r, origin_c, src.rows, src.cols));
    Mat result(src.rows, src.cols, CV_8UC1, Scalar(0));
    src.copyTo(roi);
    uchar *p;
    const short *q;
    for (int i = 0; i < result.rows; i++) {
        p = result.ptr<uchar>(i);
        for (int j = 0; j < result.cols; j++) {
            int val = 0;
            for (int k = 0; k < kernel.rows; k++) {
                q = kernel.ptr<short>(k);
                for (int l = 0; l < kernel.cols; l++) {
                    val += q[l] * big.at<uchar>(i + k, j + l);
                }
            }
            if (val > 255) p[j] = 255;
            else if (val < 0) p[j] = 0;
            else p[j] = val;
        }
    }
    return result;
}

Mat Robert(const Mat src, int thresh) {
    Mat kernel1(3, 3, CV_16S, Scalar(0));
    kernel1.at<short>(1,1) = -1;
    kernel1.at<short>(2,2) = 1;
    Mat kernel2(3, 3, CV_16S, Scalar(0));
    kernel2.at<short>(1,2) = -1;
    kernel2.at<short>(2,1) = 1;
    Mat r1, r2;
    //r1 = myConv(src, kernel1, 1, 1);
    filter2D(src, r1, CV_16S, kernel1, Point(-1,-1), 0, BORDER_DEFAULT);
    convertScaleAbs(r1, r1);
    //imwrite("r1.bmp", r1);
    //r2 = myConv(src, kernel2, 1, 1);
    filter2D(src, r2, CV_16S, kernel2, Point(-1,-1), 0, BORDER_DEFAULT);
    convertScaleAbs(r2, r2);
    //imwrite("r2.bmp", r2);
    Mat ret(src.rows, src.cols, CV_8U, Scalar(0));
    for (int i = 0; i < ret.rows; i++) {
        for (int j = 0; j < ret.cols; j++) {
            uchar v1 = r1.at<uchar>(i,j);
            uchar v2 = r2.at<uchar>(i,j);
            ret.at<uchar>(i,j) = sqrt(v1*v1 + v2*v2);
        }
    }
    threshold(ret, ret, thresh, 255, THRESH_BINARY);
    return ret;
}

Mat Prewitt(const Mat src, int thresh) {
    Mat kernel1(3, 3, CV_16S, Scalar(0));
    kernel1.at<short>(0,0) = -1;
    kernel1.at<short>(0,1) = -1;
    kernel1.at<short>(0,2) = -1;
    kernel1.at<short>(2,0) = 1;
    kernel1.at<short>(2,1) = 1;
    kernel1.at<short>(2,2) = 1;
    Mat kernel2;
    cv::transpose(kernel1, kernel2);
    Mat r1, r2;
    //r1 = myConv(src, kernel1, 1, 1);
    filter2D(src, r1, CV_16S, kernel1, Point(-1,-1), 0, BORDER_DEFAULT);
    convertScaleAbs(r1, r1);
    //imwrite("r1.bmp", r1);
    //r2 = myConv(src, kernel2, 1, 1);
    filter2D(src, r2, CV_16S, kernel2, Point(-1,-1), 0, BORDER_DEFAULT);
    convertScaleAbs(r2, r2);
    //imwrite("r2.bmp", r2);
    Mat ret(src.rows, src.cols, CV_8U, Scalar(0));
    for (int i = 0; i < ret.rows; i++) {
        for (int j = 0; j < ret.cols; j++) {
            uchar v1 = r1.at<uchar>(i,j);
            uchar v2 = r2.at<uchar>(i,j);
            ret.at<uchar>(i,j) = sqrt(v1*v1 + v2*v2);
        }
    }
    threshold(ret, ret, thresh, 255, THRESH_BINARY);
    return ret; 
}

Mat Sobel(const Mat src, int thresh) {
    Mat kernel1(3, 3, CV_16S, Scalar(0));
    kernel1.at<short>(0,0) = -1;
    kernel1.at<short>(0,1) = -2;
    kernel1.at<short>(0,2) = -1;
    kernel1.at<short>(2,0) = 1;
    kernel1.at<short>(2,1) = 2;
    kernel1.at<short>(2,2) = 1;
    Mat kernel2;
    cv::transpose(kernel1, kernel2);
    Mat r1, r2;
    //r1 = myConv(src, kernel1, 1, 1);
    filter2D(src, r1, CV_16S, kernel1, Point(-1,-1), 0, BORDER_DEFAULT);
    convertScaleAbs(r1, r1);
    //imwrite("r1.bmp", r1);
    //r2 = myConv(src, kernel2, 1, 1);
    filter2D(src, r2, CV_16S, kernel2, Point(-1,-1), 0, BORDER_DEFAULT);
    convertScaleAbs(r2, r2);
    //imwrite("r2.bmp", r2);
    Mat ret(src.rows, src.cols, CV_8U, Scalar(0));
    for (int i = 0; i < ret.rows; i++) {
        for (int j = 0; j < ret.cols; j++) {
            uchar v1 = r1.at<uchar>(i,j);
            uchar v2 = r2.at<uchar>(i,j);
            ret.at<uchar>(i,j) = sqrt(v1*v1 + v2*v2);
        }
    }
    threshold(ret, ret, thresh, 255, THRESH_BINARY);
    return ret; 
}

Mat FreiAndChen(const Mat src, int thresh) {
    Mat kernel1(3, 3, CV_16S, Scalar(0));
    kernel1.at<short>(0,0) = -1;
    kernel1.at<short>(0,1) = -1.1421;
    kernel1.at<short>(0,2) = -1;
    kernel1.at<short>(2,0) = 1;
    kernel1.at<short>(2,1) = 1.1421;
    kernel1.at<short>(2,2) = 1;
    Mat kernel2;
    cv::transpose(kernel1, kernel2);
    Mat r1, r2;
    //r1 = myConv(src, kernel1, 1, 1);
    filter2D(src, r1, CV_16S, kernel1, Point(-1,-1), 0, BORDER_DEFAULT);
    convertScaleAbs(r1, r1);
    //imwrite("r1.bmp", r1);
    //r2 = myConv(src, kernel2, 1, 1);
    filter2D(src, r2, CV_16S, kernel2, Point(-1,-1), 0, BORDER_DEFAULT);
    convertScaleAbs(r2, r2);
    //imwrite("r2.bmp", r2);
    Mat ret(src.rows, src.cols, CV_8U, Scalar(0));
    for (int i = 0; i < ret.rows; i++) {
        for (int j = 0; j < ret.cols; j++) {
            uchar v1 = r1.at<uchar>(i,j);
            uchar v2 = r2.at<uchar>(i,j);
            ret.at<uchar>(i,j) = sqrt(v1*v1 + v2*v2);
        }
    }
    threshold(ret, ret, thresh, 255, THRESH_BINARY);
    return ret; 
}

Mat Kirsch(const Mat src, int thresh) {
    Mat kernel0(3, 3, CV_16S, Scalar(0));
    kernel0.at<short>(0,0) = -3;
    kernel0.at<short>(0,1) = -3;
    kernel0.at<short>(0,2) = 5;
    kernel0.at<short>(1,0) = -3;
    kernel0.at<short>(1,2) = 5;
    kernel0.at<short>(2,0) = -3;
    kernel0.at<short>(2,1) = -3;
    kernel0.at<short>(2,2) = 5;
    Mat kernel1(3, 3, CV_16S, Scalar(0));
    kernel1.at<short>(0,0) = -3;
    kernel1.at<short>(0,1) = 5;
    kernel1.at<short>(0,2) = 5;
    kernel1.at<short>(1,0) = -3;
    kernel1.at<short>(1,2) = 5;
    kernel1.at<short>(2,0) = -3;
    kernel1.at<short>(2,1) = -3;
    kernel1.at<short>(2,2) = -3;
    Mat kernel2, kernel3, kernel4, kernel5, kernel6, kernel7;
    cv::rotate(kernel0, kernel2, ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(kernel2, kernel4, ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(kernel4, kernel6, ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(kernel1, kernel3, ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(kernel3, kernel5, ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(kernel5, kernel7, ROTATE_90_COUNTERCLOCKWISE);
    Mat k0, k1, k2, k3, k4, k5, k6, k7;
    filter2D(src, k0, CV_16S, kernel0, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k1, CV_16S, kernel1, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k2, CV_16S, kernel2, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k3, CV_16S, kernel3, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k4, CV_16S, kernel4, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k5, CV_16S, kernel5, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k6, CV_16S, kernel6, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k7, CV_16S, kernel7, Point(-1,-1), 0, BORDER_DEFAULT);
    Mat ret(src.rows, src.cols, CV_8U, Scalar(0));
    for (int i = 0; i < ret.rows; i++) {
        for (int j = 0; j < ret.cols; j++) {
            short v0 = k0.at<short>(i,j);
            short v1 = k1.at<short>(i,j);
            short v2 = k2.at<short>(i,j);
            short v3 = k3.at<short>(i,j);
            short v4 = k4.at<short>(i,j);
            short v5 = k5.at<short>(i,j);
            short v6 = k6.at<short>(i,j);
            short v7 = k7.at<short>(i,j);
            short max = std::max({v0, v1, v2, v3, v4, v5, v6, v7});
            ret.at<uchar>(i,j) = (max > thresh) ? 255 : 0;
        }
    }
    return ret; 
}

Mat Robinson(const Mat src, int thresh) {
    Mat kernel0(3, 3, CV_16S, Scalar(0));
    kernel0.at<short>(0,0) = -1;
    kernel0.at<short>(0,2) = 1;
    kernel0.at<short>(1,0) = -2;
    kernel0.at<short>(1,2) = 2;
    kernel0.at<short>(2,0) = -1;
    kernel0.at<short>(2,2) = 1;
    Mat kernel1(3, 3, CV_16S, Scalar(0));
    kernel1.at<short>(0,1) = 1;
    kernel1.at<short>(0,2) = 2;
    kernel1.at<short>(1,0) = -1;
    kernel1.at<short>(1,2) = 1;
    kernel1.at<short>(2,0) = -2;
    kernel1.at<short>(2,1) = -1;
    Mat kernel2, kernel3, kernel4, kernel5, kernel6, kernel7;
    cv::rotate(kernel0, kernel2, ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(kernel2, kernel4, ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(kernel4, kernel6, ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(kernel1, kernel3, ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(kernel3, kernel5, ROTATE_90_COUNTERCLOCKWISE);
    cv::rotate(kernel5, kernel7, ROTATE_90_COUNTERCLOCKWISE);
    Mat k0, k1, k2, k3, k4, k5, k6, k7;
    filter2D(src, k0, CV_16S, kernel0, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k1, CV_16S, kernel1, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k2, CV_16S, kernel2, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k3, CV_16S, kernel3, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k4, CV_16S, kernel4, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k5, CV_16S, kernel5, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k6, CV_16S, kernel6, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k7, CV_16S, kernel7, Point(-1,-1), 0, BORDER_DEFAULT);
    Mat ret(src.rows, src.cols, CV_8U, Scalar(0));
    for (int i = 0; i < ret.rows; i++) {
        for (int j = 0; j < ret.cols; j++) {
            short v0 = k0.at<short>(i,j);
            short v1 = k1.at<short>(i,j);
            short v2 = k2.at<short>(i,j);
            short v3 = k3.at<short>(i,j);
            short v4 = k4.at<short>(i,j);
            short v5 = k5.at<short>(i,j);
            short v6 = k6.at<short>(i,j);
            short v7 = k7.at<short>(i,j);
            short max = std::max({v0, v1, v2, v3, v4, v5, v6, v7});
            ret.at<uchar>(i,j) = (max > thresh) ? 255 : 0;
        }
    }
    return ret;
}

Mat NevatiaBabu(const Mat src, int thresh) {
    Mat kernel0(5, 5, CV_16S, Scalar(-100));
    kernel0.at<short>(0,0) = 100;
    kernel0.at<short>(0,1) = 100;
    kernel0.at<short>(0,2) = 100;
    kernel0.at<short>(0,3) = 100;
    kernel0.at<short>(0,4) = 100;
    kernel0.at<short>(1,0) = 100;
    kernel0.at<short>(1,1) = 100;
    kernel0.at<short>(1,2) = 100;
    kernel0.at<short>(1,3) = 100;
    kernel0.at<short>(1,4) = 100;
    kernel0.at<short>(2,0) = 0;
    kernel0.at<short>(2,1) = 0;
    kernel0.at<short>(2,2) = 0;
    kernel0.at<short>(2,3) = 0;
    kernel0.at<short>(2,4) = 0;
    Mat kernel1(5, 5, CV_16S, Scalar(-100));
    kernel1.at<short>(0,0) = 100;
    kernel1.at<short>(0,1) = 100;
    kernel1.at<short>(0,2) = 100;
    kernel1.at<short>(0,3) = 100;
    kernel1.at<short>(0,4) = 100;
    kernel1.at<short>(1,0) = 100;
    kernel1.at<short>(1,1) = 100;
    kernel1.at<short>(1,2) = 100;
    kernel1.at<short>(1,3) = 78;
    kernel1.at<short>(1,4) = -32;
    kernel1.at<short>(2,0) = 100;
    kernel1.at<short>(2,1) = 92;
    kernel1.at<short>(2,2) = 0;
    kernel1.at<short>(2,3) = -92;
    kernel1.at<short>(2,4) = -100;
    kernel1.at<short>(3,0) = 32;
    kernel1.at<short>(3,1) = -78;
    kernel1.at<short>(3,2) = -100;
    kernel1.at<short>(3,3) = -100;
    kernel1.at<short>(3,4) = -100;
    Mat kernel2(5, 5, CV_16S, Scalar(-100));
    kernel2.at<short>(0,0) = 100;
    kernel2.at<short>(0,1) = 100;
    kernel2.at<short>(0,2) = 100;
    kernel2.at<short>(0,3) = 32;
    kernel2.at<short>(1,0) = 100;
    kernel2.at<short>(1,1) = 100;
    kernel2.at<short>(1,2) = 92;
    kernel2.at<short>(1,3) = -78;
    kernel2.at<short>(2,0) = 100;
    kernel2.at<short>(2,1) = 100;
    kernel2.at<short>(2,2) = 0;
    kernel2.at<short>(2,3) = -100;
    kernel2.at<short>(3,0) = 100;
    kernel2.at<short>(3,1) = 78;
    kernel2.at<short>(3,2) = -92;
    kernel2.at<short>(3,3) = -100;
    kernel2.at<short>(4,0) = 100;
    kernel2.at<short>(4,1) = -32;
    kernel2.at<short>(4,2) = -100;
    kernel2.at<short>(4,3) = -100;
    Mat kernel3, kernel4, kernel5;
    cv::rotate(kernel0, kernel3, ROTATE_90_CLOCKWISE);
    cv::rotate(kernel1, kernel4, ROTATE_90_CLOCKWISE);
    cv::rotate(kernel2, kernel5, ROTATE_90_CLOCKWISE);
    Mat k0, k1, k2, k3, k4, k5;
    filter2D(src, k0, CV_16S, kernel0, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k1, CV_16S, kernel1, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k2, CV_16S, kernel2, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k3, CV_16S, kernel3, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k4, CV_16S, kernel4, Point(-1,-1), 0, BORDER_DEFAULT);
    filter2D(src, k5, CV_16S, kernel5, Point(-1,-1), 0, BORDER_DEFAULT);
    Mat ret(src.rows, src.cols, CV_8U, Scalar(0));
    for (int i = 0; i < ret.rows; i++) {
        for (int j = 0; j < ret.cols; j++) {
            short v0 = k0.at<short>(i,j);
            short v1 = k1.at<short>(i,j);
            short v2 = k2.at<short>(i,j);
            short v3 = k3.at<short>(i,j);
            short v4 = k4.at<short>(i,j);
            short v5 = k5.at<short>(i,j);
            short max = std::max({v0, v1, v2, v3, v4, v5});
            ret.at<uchar>(i,j) = (max > thresh) ? 255 : 0;
        }
    }
    return ret;
}

int main(int argc, char** argv) {
    Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    CV_Assert(img_gray.depth() == CV_8U);
    int channels = img_gray.channels();
    int nRows = img_gray.rows;
    int nCols = img_gray.cols * channels;
    imwrite("robert.bmp", Robert(img_gray, 24));
    imwrite("prewitt.bmp", Prewitt(img_gray, 70));
    imwrite("sobel.bmp", Sobel(img_gray, 70));
    imwrite("frei_and_chen.bmp", FreiAndChen(img_gray, 70));
    imwrite("kirsch.bmp", Kirsch(img_gray, 250));
    imwrite("robinson.bmp", Robinson(img_gray, 100));
    imwrite("nevatia_babu.bmp", NevatiaBabu(img_gray, 17500));
}