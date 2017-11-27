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

enum YokoiType {
    Yokoi_q, Yokoi_r, Yokoi_s, Yokoi_init
};

enum BIType {//border interior
    BI_background, BI_border, BI_interior
};

enum PRType {//pair relationship
    PR_background, PR_p, PR_q
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

void PrintSymbloic(const Mat src) {
    for (int i = 0; i < src.rows; i++) {
        const uchar *p = src.ptr<uchar>(i);
        for (int j = 0; j < src.cols; j++) {
            if (p[j] == 0) {
                cout << " ";
            } else {
                cout << (int)p[j];
            }
        }
        cout << endl;
    }
}

void DecideYokoiType(YokoiType& a, uchar b, uchar c, uchar d, uchar e) {//follow powerpoint
    if (b == c && (d != b || e != b)) a = Yokoi_q;
    else if (b == c && (d == b && e == b)) a = Yokoi_r;
    else if (b != c) a = Yokoi_s;
}

void DecideYokoiType8(YokoiType& a, uchar b, uchar c, uchar d, uchar e) {//follow powerpoint
    if (b != c && (d == b || e == b)) a = Yokoi_q;
    else if (b == c && (d == b && e == b)) a = Yokoi_r;
    else if (b != c) a = Yokoi_s;
}

Mat InteriorBorder(const Mat src, int connect_type) {//connect_type = 4 or 8
    Mat result(src.rows, src.cols, CV_8UC1, Scalar(0));//has padding
    if (connect_type == 4) {
        for (int i = 1; i < src.rows - 1; i++) {
            for (int j = 1; j < src.cols - 1; j++) {
                if (src.at<uchar>(i,j) != 255) continue;
                int a0 = 255;
                int a1 = (a0 == src.at<uchar>(i,j + 1)) ? a0 : BI_border;
                int a2 = (a1 == src.at<uchar>(i - 1,j)) ? a1 : BI_border;
                int a3 = (a2 == src.at<uchar>(i,j - 1)) ? a2 : BI_border;
                int a4 = (a3 == src.at<uchar>(i + 1,j)) ? a3 : BI_border;
                if (a4 == 1) {
                    result.at<uchar>(i,j) = BI_border;
                } else {
                    result.at<uchar>(i,j) = BI_interior;
                }
            }
        }
    } else if (connect_type == 8) {
        for (int i = 1; i < src.rows - 1; i++) {
            for (int j = 1; j < src.cols - 1; j++) {
                if (src.at<uchar>(i,j) != 255) continue;
                int a0 = 255;
                int a1 = (a0 == src.at<uchar>(i,j + 1)) ? a0 : BI_border;
                int a2 = (a1 == src.at<uchar>(i - 1,j)) ? a1 : BI_border;
                int a3 = (a2 == src.at<uchar>(i,j - 1)) ? a2 : BI_border;
                int a4 = (a3 == src.at<uchar>(i + 1,j)) ? a3 : BI_border;
                int a5 = (a3 == src.at<uchar>(i + 1,j + 1)) ? a4 : BI_border;
                int a6 = (a3 == src.at<uchar>(i - 1,j + 1)) ? a5 : BI_border;
                int a7 = (a3 == src.at<uchar>(i - 1,j - 1)) ? a6 : BI_border;
                int a8 = (a3 == src.at<uchar>(i + 1,j - 1)) ? a7 : BI_border;
                if (a8 == 1) {
                    result.at<uchar>(i,j) = BI_border;
                } else {
                    result.at<uchar>(i,j) = BI_interior;
                }
            }
        }
    }
    return result;
}

Mat PairRelationship(const Mat src, int connect_type) {//connect_type = 4 or 8
    Mat result(src.rows, src.cols, CV_8UC1, Scalar(0));//has padding
    if (connect_type == 4) {
        for (int i = 1; i < src.rows - 1; i++) {
            for (int j = 1; j < src.cols - 1; j++) {
                if (src.at<uchar>(i,j) == 0) continue;
                int a0 = src.at<uchar>(i,j);
                int a1 = (src.at<uchar>(i,j + 1) == BI_interior) ? 1 : 0;
                int a2 = (src.at<uchar>(i - 1, j) == BI_interior) ? 1 : 0;
                int a3 = (src.at<uchar>(i,j - 1) == BI_interior) ? 1 : 0;
                int a4 = (src.at<uchar>(i + 1,j) == BI_interior) ? 1 : 0;
                if (a0 == BI_border && (a1 + a2 + a3 + a4) >= 1) {
                    result.at<uchar>(i,j) = PR_p;
                } else {
                    result.at<uchar>(i,j) = PR_q;
                }
            }
        }
    } else if (connect_type == 8) {
        for (int i = 1; i < src.rows - 1; i++) {
            for (int j = 1; j < src.cols - 1; j++) {
                if (src.at<uchar>(i,j) == 0) continue;
                int a0 = src.at<uchar>(i,j);
                int a1 = (src.at<uchar>(i,j + 1) == BI_interior) ? 1 : 0;
                int a2 = (src.at<uchar>(i - 1, j) == BI_interior) ? 1 : 0;
                int a3 = (src.at<uchar>(i,j - 1) == BI_interior) ? 1 : 0;
                int a4 = (src.at<uchar>(i + 1,j) == BI_interior) ? 1 : 0;
                int a5 = (src.at<uchar>(i + 1,j + 1) == BI_interior) ? 1 : 0;
                int a6 = (src.at<uchar>(i - 1,j + 1) == BI_interior) ? 1 : 0;
                int a7 = (src.at<uchar>(i - 1,j - 1) == BI_interior) ? 1 : 0;
                int a8 = (src.at<uchar>(i + 1,j - 1) == BI_interior) ? 1 : 0;
                if (a0 == BI_border && (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8) >= 1) {
                    result.at<uchar>(i,j) = PR_p;
                } else {
                    result.at<uchar>(i,j) = PR_q;
                }
            }
        }
    }
    return result;
}

Mat Thinning(const Mat src, const Mat pair_relationship, int connect_type) {
    Mat result = src.clone();
    if (connect_type == 4) {
        for (int i = 1; i < src.rows - 1; i++) {
            for (int j = 1; j < src.cols - 1; j++) {
                if (pair_relationship.at<uchar>(i,j) != PR_p) continue;
                YokoiType a1 = Yokoi_init, a2 = Yokoi_init, a3 = Yokoi_init, a4 = Yokoi_init;
                DecideYokoiType(a1, result.at<uchar>(i,j), result.at<uchar>(i,j+1), result.at<uchar>(i-1,j+1), result.at<uchar>(i-1,j));
                DecideYokoiType(a2, result.at<uchar>(i,j), result.at<uchar>(i-1,j), result.at<uchar>(i-1,j-1), result.at<uchar>(i,j-1));
                DecideYokoiType(a3, result.at<uchar>(i,j), result.at<uchar>(i,j-1), result.at<uchar>(i+1,j-1), result.at<uchar>(i+1,j));
                DecideYokoiType(a4, result.at<uchar>(i,j), result.at<uchar>(i+1,j), result.at<uchar>(i+1,j+1), result.at<uchar>(i,j+1));
                int temp = 0;
                if (a1 == Yokoi_q) temp +=1;
                if (a2 == Yokoi_q) temp +=1;
                if (a3 == Yokoi_q) temp +=1;
                if (a4 == Yokoi_q) temp +=1;
                if (temp == 1) result.at<uchar>(i,j) = 0;
            }
        }
    } else if (connect_type == 8) {
        for (int i = 1; i < src.rows - 1; i++) {
            for (int j = 1; j < src.cols - 1; j++) {
                if (pair_relationship.at<uchar>(i,j) != PR_p) continue;
                YokoiType a1 = Yokoi_init, a2 = Yokoi_init, a3 = Yokoi_init, a4 = Yokoi_init;
                DecideYokoiType8(a1, result.at<uchar>(i,j), result.at<uchar>(i,j+1), result.at<uchar>(i-1,j+1), result.at<uchar>(i-1,j));
                DecideYokoiType8(a2, result.at<uchar>(i,j), result.at<uchar>(i-1,j), result.at<uchar>(i-1,j-1), result.at<uchar>(i,j-1));
                DecideYokoiType8(a3, result.at<uchar>(i,j), result.at<uchar>(i,j-1), result.at<uchar>(i+1,j-1), result.at<uchar>(i+1,j));
                DecideYokoiType8(a4, result.at<uchar>(i,j), result.at<uchar>(i+1,j), result.at<uchar>(i+1,j+1), result.at<uchar>(i,j+1));
                int temp = 0;
                if (a1 == Yokoi_q) temp +=1;
                if (a2 == Yokoi_q) temp +=1;
                if (a3 == Yokoi_q) temp +=1;
                if (a4 == Yokoi_q) temp +=1;
                if (temp == 1) result.at<uchar>(i,j) = 0;
            }
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
    Mat pad(nRows/8 + 2, nCols/8 + 2, CV_8UC1, Scalar(0));
    Mat roi(pad, Rect(1, 1, nRows/8, nCols/8));
    Mat small = DownSample(img_binary, 8);
    small.copyTo(roi);
    bool eq = false;
    Mat origin_img = pad.clone();
    /*
    Mat origin_img(8,9,CV_8UC1, Scalar(0));
    origin_img.at<uchar>(1,1) = 255;
    origin_img.at<uchar>(1,2) = 255;
    origin_img.at<uchar>(1,3) = 255;
    origin_img.at<uchar>(1,4) = 255;
    origin_img.at<uchar>(1,5) = 255;
    origin_img.at<uchar>(1,6) = 255;
    origin_img.at<uchar>(2,1) = 255;
    origin_img.at<uchar>(2,2) = 255;
    origin_img.at<uchar>(2,3) = 255;
    origin_img.at<uchar>(2,4) = 255;
    origin_img.at<uchar>(2,5) = 255;
    origin_img.at<uchar>(2,6) = 255;
    origin_img.at<uchar>(3,1) = 255;
    origin_img.at<uchar>(3,2) = 255;
    origin_img.at<uchar>(3,3) = 255;
    origin_img.at<uchar>(3,4) = 255;
    origin_img.at<uchar>(3,5) = 255;
    origin_img.at<uchar>(3,6) = 255;
    origin_img.at<uchar>(3,7) = 255;
    origin_img.at<uchar>(4,1) = 255;
    origin_img.at<uchar>(4,4) = 255;
    origin_img.at<uchar>(4,5) = 255;
    origin_img.at<uchar>(4,6) = 255;
    origin_img.at<uchar>(4,7) = 255;
    origin_img.at<uchar>(5,1) = 255;
    origin_img.at<uchar>(5,5) = 255;
    origin_img.at<uchar>(5,6) = 255;
    origin_img.at<uchar>(5,7) = 255;
    origin_img.at<uchar>(6,1) = 255;
    origin_img.at<uchar>(6,7) = 255;*/
    Mat output_img;
    while (!eq) {
        Mat interior_border = InteriorBorder(origin_img, 8);
        Mat pair_relationship = PairRelationship(interior_border, 8);
        //PrintSymbloic(interior_border);
        //PrintSymbloic(pair_relationship);
        output_img = Thinning(origin_img, pair_relationship, 4);
        Mat diff = origin_img != output_img;
        eq = (cv::countNonZero(diff) == 0);
        if (!eq) origin_img = output_img.clone();
    }
    imwrite("thin.bmp", output_img);
}