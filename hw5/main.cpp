#include <iostream>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <ctime>
using namespace std;
using namespace cv;
//#define DEBUG
#ifdef DEBUG
    #define D(x) x
#else
    #define D(x)
#endif
//same as formula
Mat Dilation(const Mat source, const Mat structure, int origin_r, int origin_c) {//gray image
    int channels = source.channels();
    int nRows = source.rows;
    int nCols = source.cols * channels;
    Mat big(nRows + structure.rows - 1, nCols + structure.cols - 1, CV_8UC1, Scalar(0));
    Mat roi(big, Rect(structure.rows - origin_r - 1, structure.cols - origin_c - 1, nRows, nCols));
    Mat result(nRows, nCols, CV_8UC1, Scalar(0));
    source.copyTo(roi);
    uchar *p;
    const uchar *q;
    for (int i = 0; i < nRows; i++) {
        p = result.ptr<uchar>(i);
        for (int j = 0; j < nCols; j++) {
            int max = 0;
            for (int k = 0; k < structure.rows; k++) {
                q = structure.ptr<uchar>(k);
                for (int l = 0; l < structure.cols; l++) {
                    int temp = big.at<uchar>(i + structure.rows - 1 - k, j + structure.cols - 1 - l) + q[l];
                    if (max < temp) max = temp;
                }
            }
            p[j] = max;
        }
    }    
    return result;
}
//my interpretation, faster
Mat Dilation2(const Mat source, const Mat structure, int origin_r, int origin_c) {//gray image
    int channels = source.channels();
    int nRows = source.rows;
    int nCols = source.cols * channels;
    Mat big(nRows + structure.rows - 1, nCols + structure.cols - 1, CV_8UC1, Scalar(0));
    const uchar *p, *q;
    for (int i = 0; i < nRows; i++) {
        p = source.ptr<uchar>(i);
        for (int j = 0; j < nCols; j++) {
            for (int k = 0; k < structure.rows; k++) {
                q = structure.ptr<uchar>(k);
                for (int l = 0; l < structure.cols; l++) {
                    int temp = p[j] + q[l];
                    if (temp > big.at<uchar>(i + k, j + l)) {
                        big.at<uchar>(i + k, j + l) = temp;
                    }
                }
            }
        }
    }
    Mat result(big, Rect(origin_r, origin_c, nRows, nCols));
    return result;
}

Mat Erosion(const Mat source, const Mat structure, int origin_r, int origin_c) {//gray image
    int channels = source.channels();
    int nRows = source.rows;
    int nCols = source.cols * channels;
    Mat big(nRows + structure.rows - 1, nCols + structure.cols - 1, CV_8UC1, Scalar(255));
    Mat roi(big, Rect(origin_r, origin_c, nRows, nCols));
    Mat result(nRows, nCols, CV_8UC1, Scalar(255));
    source.copyTo(roi);
    uchar *p;
    const uchar *q;
    for (int i = 0; i < nRows; i++) {
        p = result.ptr<uchar>(i);
        for (int j = 0; j < nCols; j++) {
            int min = 255;
            for (int k = 0; k < structure.rows; k++) {
                q = structure.ptr<uchar>(k);
                for (int l = 0; l < structure.cols; l++) {
                   int temp = big.at<uchar>(i + k, j + l) - q[l];
                   if (min > temp) min = temp;
                }
            }
            p[j] = min;
        }
    }
    return result;
}

int main(int argc, char** argv) {
    Mat img_binary = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    CV_Assert(img_binary.depth() == CV_8U);
    int channels = img_binary.channels();
    int nRows = img_binary.rows;
    int nCols = img_binary.cols * channels;

    Mat circle_structure(5, 5, CV_8UC1, Scalar(0));
    circle_structure.at<uchar>(0, 1) = 0;
    circle_structure.at<uchar>(0, 2) = 0;
    circle_structure.at<uchar>(0, 3) = 0;
    circle_structure.at<uchar>(1, 0) = 0;
    circle_structure.at<uchar>(1, 1) = 0;
    circle_structure.at<uchar>(1, 2) = 0;
    circle_structure.at<uchar>(1, 3) = 0;
    circle_structure.at<uchar>(1, 4) = 0;
    circle_structure.at<uchar>(2, 0) = 0;
    circle_structure.at<uchar>(2, 1) = 0;
    circle_structure.at<uchar>(2, 2) = 0;
    circle_structure.at<uchar>(2, 3) = 0;
    circle_structure.at<uchar>(2, 4) = 0;
    circle_structure.at<uchar>(3, 0) = 0;
    circle_structure.at<uchar>(3, 1) = 0;
    circle_structure.at<uchar>(3, 2) = 0;
    circle_structure.at<uchar>(3, 3) = 0;
    circle_structure.at<uchar>(3, 4) = 0;
    circle_structure.at<uchar>(4, 1) = 0;
    circle_structure.at<uchar>(4, 2) = 0;
    circle_structure.at<uchar>(4, 3) = 0;
    //dilation
    //std::clock_t    start;
    //start = std::clock();
    Mat dilation = Dilation(img_binary, circle_structure, 2, 2);
    //std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    imwrite("dilation.bmp", dilation);
    /*
    //start = std::clock();
    Mat dilation2 = Dilation2(img_binary, circle_structure, 2, 2);
    //std::cout << "Time: " << (std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    imwrite("dilation2.bmp", dilation2);
    */
    //erosion
    Mat erosion = Erosion(img_binary, circle_structure, 2, 2);
    imwrite("erosion.bmp", erosion);
    //opening
    Mat opening = Dilation(Erosion(img_binary, circle_structure, 2, 2), circle_structure, 2, 2);
    imwrite("opening.bmp", opening);
    //closing
    Mat closing = Erosion(Dilation(img_binary, circle_structure, 2, 2), circle_structure, 2, 2);
    imwrite("closing.bmp", closing);
}