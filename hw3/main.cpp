#include <iostream>
#include <opencv2/opencv.hpp>
#include <numeric>
using namespace std;
using namespace cv;
//#define DEBUG
#ifdef DEBUG
    #define D(x) x
#else
    #define D(x)
#endif

int main(int argc, char** argv) {
    Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    Mat img_out;
    int histogram_array[256] = {};
    int cdf_array[256] = {};
    //create look-up table
    CV_Assert(img_gray.depth() == CV_8U);
    int channels = img_gray.channels();
    int nRows = img_gray.rows;
    int nCols = img_gray.cols * channels;

    if (img_gray.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }

    int i,j;
    uchar* p;
    for( i = 0; i < nRows; ++i) {
        p = img_gray.ptr<uchar>(i);
        for ( j = 0; j < nCols; ++j) {
            histogram_array[p[j]] += 1;
        }
    }
    std::partial_sum(histogram_array, histogram_array+ 256, cdf_array);
    std::for_each(cdf_array, cdf_array+256, [nRows, nCols](int& x) {x = 255*x / (nRows * nCols);});
    Mat lookUpTable(1, 256, CV_8U);
    p = lookUpTable.data;
    for( int i = 0; i < 256; ++i)
        p[i] = cdf_array[i];
    //apply look-up table
    LUT(img_gray, lookUpTable, img_out);
    imwrite("equalization.bmp", img_out);
    //create histogram
    Mat histogram_img(256,256,CV_8UC1,Scalar(255));
    CV_Assert(img_out.depth() == CV_8U);
    channels = img_out.channels();
    nRows = img_out.rows;
    nCols = img_out.cols * channels;
    std::for_each(histogram_array, histogram_array+256, [](int& x) {x = 0;});
    if (img_out.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }
    for( i = 0; i < nRows; ++i) {
        p = img_out.ptr<uchar>(i);
        for ( j = 0; j < nCols; ++j) {
            histogram_array[p[j]] += 1;
        }
    }
    float scale = 0.95 * 256 / (*max_element(histogram_array, histogram_array + 256));
    for (int i = 0; i < 256; i++) {
        int intensity = (int)(scale * histogram_array[i]);
        line(histogram_img, Point(i,255), Point(i,255-intensity), Scalar(0));
    }
    imwrite("histogram.bmp", histogram_img);
}