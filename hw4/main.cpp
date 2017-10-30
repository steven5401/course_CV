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
Mat Dilation(const Mat source, const Mat structure, int origin_r, int origin_c) {//gray image
    int channels = source.channels();
    int nRows = source.rows;
    int nCols = source.cols * channels;
    Mat big(nRows + structure.rows - 1, nCols + structure.cols - 1, CV_8UC1, Scalar(0));
    const uchar *p, *q;
    for (int i = 0; i < nRows; i++) {
        p = source.ptr<uchar>(i);
        for (int j = 0; j < nCols; j++) {
            if (p[j] == 255) {
                for (int k = 0; k < structure.rows; k++) {
                    q = structure.ptr<uchar>(k);
                    for (int l = 0; l < structure.cols; l++) {
                        if (q[l] == 255) {
                            big.at<uchar>(i + k, j + l) = 255;
                        }
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
    Mat big(nRows + structure.rows - 1, nCols + structure.cols - 1, CV_8UC1, Scalar(0));
    Mat result(nRows, nCols, CV_8UC1, Scalar(255));
    const uchar *p, *q;
    uchar *r;
    for (int i = 0; i < nRows; i++) {
        p = source.ptr<uchar>(i);
        for (int j = 0; j < nCols; j++) {
           big.at<uchar>(i + origin_r, j + origin_c) = p[j];
        }
    }
    for (int i = 0; i < nRows; i++) {
        r = result.ptr<uchar>(i);
        for (int j = 0; j < nCols; j++) {
            for (int k = 0; k < structure.rows; k++) {
                bool flag = false;//break outside structure scan
                q = structure.ptr<uchar>(k);
                for (int l = 0; l < structure.cols; l++) {
                    if (q[l] == 255 && big.at<uchar>(i + k, j + l) != 255) {
                        r[j] = 0;
                        flag = true;
                        break;
                    }
                }
                if (flag) break;
            }
        }
    }
    return result;
}

Mat MatUnion(const Mat a, const Mat b) {//a and b should be same size
    CV_Assert(a.size == b.size);
    Mat result(a.rows, a.cols, CV_8UC1, Scalar(0));
    int channels = a.channels();
    int nRows = a.rows;
    int nCols = a.cols * channels;
    const uchar *p, *q;
    uchar *r;
    for (int i = 0; i < nRows; i++) {
        p = a.ptr<uchar>(i);
        q = b.ptr<uchar>(i);
        r = result.ptr<uchar>(i);
        for (int j = 0; j < nCols; j++) {
            if (p[j] == 255 && q[j] == 255) r[j] = 255;
        }
    }
    return result;
}

Mat MatComplement(const Mat source) {
    Mat result(source.rows, source.cols, CV_8UC1, Scalar(0));
    int channels = source.channels();
    int nRows = source.rows;
    int nCols = source.cols * channels;
    const uchar *p;
    uchar *r;
    for (int i = 0; i < nRows; i++) {
        p = source.ptr<uchar>(i);
        r = result.ptr<uchar>(i);
        for (int j = 0; j < nCols; j++) {
            if (p[j] == 0) r[j] = 255;
        }
    }
    return result;
}

Mat HitAndMiss(const Mat source, const Mat j, const Mat k, int originj_r, int originj_c, int origink_r, int origink_c) {
    Mat a = Erosion(source, j, originj_r, originj_c);
    Mat b = Erosion(MatComplement(source), k, origink_r, origink_c);
    return MatUnion(a, b);
}

int main(int argc, char** argv) {
    Mat img_binary = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    CV_Assert(img_binary.depth() == CV_8U);
    int channels = img_binary.channels();
    int nRows = img_binary.rows;
    int nCols = img_binary.cols * channels;
    //to binary
    uchar *p;
    for (int i = 0; i < nRows; ++i) {
        p = img_binary.ptr<uchar>(i);
        for (int j = 0; j < nCols; ++j) {
            if (p[j] >= 128) {
                p[j] = 255;
            } else {
                p[j] = 0;
            }
        }
    }
    //dilation
    //structure element {(0,0), (-1,0), (0,-1), (1,0), (0,1)}
    /*Mat dilation_structure(3, 3, CV_8UC1, Scalar(0));
    dilation_structure.at<uchar>(0, 1) = 255;
    dilation_structure.at<uchar>(1, 0) = 255;
    dilation_structure.at<uchar>(1, 1) = 255;
    dilation_structure.at<uchar>(1, 2) = 255;
    dilation_structure.at<uchar>(2, 1) = 255;*/
    Mat circle_structure(5, 5, CV_8UC1, Scalar(0));
    circle_structure.at<uchar>(0, 1) = 255;
    circle_structure.at<uchar>(0, 2) = 255;
    circle_structure.at<uchar>(0, 3) = 255;
    circle_structure.at<uchar>(1, 0) = 255;
    circle_structure.at<uchar>(1, 1) = 255;
    circle_structure.at<uchar>(1, 2) = 255;
    circle_structure.at<uchar>(1, 3) = 255;
    circle_structure.at<uchar>(1, 4) = 255;
    circle_structure.at<uchar>(2, 0) = 255;
    circle_structure.at<uchar>(2, 1) = 255;
    circle_structure.at<uchar>(2, 2) = 255;
    circle_structure.at<uchar>(2, 3) = 255;
    circle_structure.at<uchar>(2, 4) = 255;
    circle_structure.at<uchar>(3, 0) = 255;
    circle_structure.at<uchar>(3, 1) = 255;
    circle_structure.at<uchar>(3, 2) = 255;
    circle_structure.at<uchar>(3, 3) = 255;
    circle_structure.at<uchar>(3, 4) = 255;
    circle_structure.at<uchar>(4, 1) = 255;
    circle_structure.at<uchar>(4, 2) = 255;
    circle_structure.at<uchar>(4, 3) = 255;

    Mat dilation = Dilation(img_binary, circle_structure, 2, 2);
    imwrite("dilation.bmp", dilation);
    //erosion
    //structure element is same as dilation
    Mat erosion = Erosion(img_binary, circle_structure, 2, 2);
    imwrite("erosion.bmp", erosion);
    //opening
    Mat opening = Dilation(Erosion(img_binary, circle_structure, 2, 2), circle_structure, 2, 2);
    imwrite("opening.bmp", opening);
    //closing
    Mat closing = Erosion(Dilation(img_binary, circle_structure, 2, 2), circle_structure, 2, 2);
    imwrite("closing.bmp", closing);
    //hit-and-miss
    Mat structure(2, 2, CV_8UC1, Scalar(255));
    structure.at<uchar>(1,0) = 0;
    Mat hit_and_miss = HitAndMiss(img_binary, structure, structure, 0, 1, 1, 0);
    imwrite("hit-and-miss.bmp", hit_and_miss);
}