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
enum YokoiType {
    q, r, s, init
};

Mat DownSample(const Mat src, int block_size) {
    //take the topmost-left pixel in each block as the downsampled data.
    int output_r = src.rows / block_size;
    int output_c = src.cols / block_size;
    Mat output(output_r, output_c, CV_8UC1, Scalar(0));
    uchar *p;
    for (int i = 0; i < output_r; ++i) {
        p = output.ptr<uchar>(i);
        for (int j = 0; j < output_c; ++j) {
            p[j] = src.at<uchar>(block_size * i, block_size * j);
        }
    }
    return output;
}

void DecideYokoiType(YokoiType& a, uchar b, uchar c, uchar d, uchar e) {//follow powerpoint
    if (b == c && (d != b || e != b)) a = q;
    else if (b == c && (d == b && e == b)) a = r;
    else if (b != c) a = s;
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
    Mat yokoi(nRows/8, nCols/8, CV_8UC1, Scalar(0));
    Mat pad(nRows/8 + 2, nCols/8 + 2, CV_8UC1, Scalar(0));
    Mat roi(pad, Rect(1, 1, nRows/8, nCols/8));
    Mat small = DownSample(img_binary, 8);
    small.copyTo(roi);
    for (int i = 1; i < pad.rows - 1; i++) {
        for (int j = 1; j < pad.cols - 1; j++) {
            if (pad.at<uchar>(i,j) != 255) continue;
            YokoiType a1 = init, a2 = init, a3 = init, a4 = init;
            DecideYokoiType(a1, pad.at<uchar>(i,j), pad.at<uchar>(i,j+1), pad.at<uchar>(i-1,j+1), pad.at<uchar>(i-1,j));
            DecideYokoiType(a2, pad.at<uchar>(i,j), pad.at<uchar>(i-1,j), pad.at<uchar>(i-1,j-1), pad.at<uchar>(i,j-1));
            DecideYokoiType(a3, pad.at<uchar>(i,j), pad.at<uchar>(i,j-1), pad.at<uchar>(i+1,j-1), pad.at<uchar>(i+1,j));
            DecideYokoiType(a4, pad.at<uchar>(i,j), pad.at<uchar>(i+1,j), pad.at<uchar>(i+1,j+1), pad.at<uchar>(i,j+1));
            if (a1 == r && a2 == r && a3 == r && a4 == r) {
                yokoi.at<uchar>(i-1,j-1) = 5;
            } else {
                int temp = 0;
                if (a1 == q) temp +=1;
                if (a2 == q) temp +=1;
                if (a3 == q) temp +=1;
                if (a4 == q) temp +=1;
                yokoi.at<uchar>(i-1,j-1) = temp;
            }
        }
    }
    for (int i = 0; i < yokoi.rows; i++) {
        const uchar *p = yokoi.ptr<uchar>(i);
        for (int j = 0; j < yokoi.cols; j++) {
            if (p[j] == 0) {
                cout << " ";
            } else {
                cout << (int)p[j];
            }
        }
        cout << endl;
    }
}