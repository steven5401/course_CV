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

Mat Opening(const Mat source, const Mat structure, int origin_r, int origin_c) {
    return Dilation(Erosion(source, structure, origin_r, origin_c), structure, origin_r, origin_c);
}

Mat Closing(const Mat source, const Mat structure, int origin_r, int origin_c) {
    return Erosion(Dilation(source, structure, origin_r, origin_c), structure, origin_r, origin_c);
}

Mat AddGaussianNoise(const Mat src, double amplitude) {
    Mat ret = src.clone();
    int nRows = ret.rows;
    int nCols = ret.cols * ret.channels();
    default_random_engine generator;
    normal_distribution<double> distribution (0.0,1.0);
    uchar* p;
    for(int i = 0; i < nRows; ++i) {
        p = ret.ptr<uchar>(i);
        for (int j = 0; j < nCols; ++j) {
            double number = distribution(generator);
            int r = p[j] + amplitude * number;
            p[j] = (0 <= r && r <= 255) ? r : p[j];
        }
    }
    return ret;
}

Mat AddSaltAndPepperNoise(const Mat src, double threshold) {
    Mat ret = src.clone();
    int nRows = ret.rows;
    int nCols = ret.cols * ret.channels();
    default_random_engine generator;
    uniform_real_distribution<double> distribution(0.0,1.0);
    uchar* p;
    for(int i = 0; i < nRows; ++i) {
        p = ret.ptr<uchar>(i);
        for (int j = 0; j < nCols; ++j) {
            double number = distribution(generator);
            if (number < threshold) {
                p[j] = 0;
            } else if (number > 1.0 - threshold) {
                p[j] = 255;
            } else {
                //do nothing
            }
        }
    }
    return ret;
}

Mat BoxFilter(const Mat src, int size) {
    int nRows = src.rows;
    int nCols = src.cols * src.channels();
    Mat big(nRows + size - 1, nCols + size - 1, CV_8UC1, Scalar(0));
    Mat roi(big, Rect((size - 1) / 2, (size - 1) / 2, nRows, nCols));
    Mat ret(nRows, nCols, CV_8UC1, Scalar(0));
    src.copyTo(roi);
    uchar *p;
    int denominator = size * size;
    for (int i = 0; i < nRows; i++) {
        p = ret.ptr<uchar>(i);
        for (int j = 0; j < nCols; j++) {
            int sum = 0;
            for (int k = 0; k < size; k++) {
                for (int l = 0 ; l < size; l++) {
                    sum += big.at<uchar>(i + k, j + l);
                }
            }
            p[j] = sum / denominator;
        }
    }
    return ret;
}

Mat MedianFilter(const Mat src, int size) {
    int nRows = src.rows;
    int nCols = src.cols * src.channels();
    Mat big(nRows + size - 1, nCols + size - 1, CV_8UC1, Scalar(0));
    Mat roi(big, Rect((size - 1) / 2, (size - 1) / 2, nRows, nCols));
    Mat ret(nRows, nCols, CV_8UC1, Scalar(0));
    src.copyTo(roi);
    uchar *p;
    for (int i = 0; i < nRows; i++) {
        p = ret.ptr<uchar>(i);
        for (int j = 0; j < nCols; j++) {
            int* value_list = new int[size*size];
            int idx = 0;
            for (int k = 0; k < size; k++) {
                for (int l = 0 ; l < size; l++) {
                    value_list[idx] = big.at<uchar>(i + k,j + l);
                    idx++;
                }
            }
            nth_element(value_list, value_list + (size*size - 1)/2, value_list + size*size);
            p[j] = value_list[(size*size - 1)/2];
            delete[] value_list;
        }
    }
    return ret;
}

double SNR(const Mat src, const Mat noise) {//two Mat must be same size
    int nRows = src.rows;
    int nCols = src.cols * src.channels();
    int size = nRows * nCols;
    const uchar *p, *q;
    double src_sum = 0;
    double noise_sum = 0;
    for(int i = 0; i < nRows; ++i) {
        p = src.ptr<uchar>(i);
        q = noise.ptr<uchar>(i);
        for (int j = 0; j < nCols; ++j) {
            src_sum += p[j];
            noise_sum += (q[j] - p[j]);
        }
    }
    double src_mean = src_sum / size;
    double noise_mean = noise_sum / size;
    src_sum = 0;
    noise_sum = 0;
    for(int i = 0; i < nRows; ++i) {
        p = src.ptr<uchar>(i);
        q = noise.ptr<uchar>(i);
        for (int j = 0; j < nCols; ++j) {
            src_sum += ((p[j] - src_mean) * (p[j] - src_mean));
            noise_sum += ((q[j] - p[j] - noise_mean) * (q[j] - p[j] - noise_mean));
        }
    }
    double src_variance = src_sum / size;
    double noise_variance = noise_sum / size;
    double snr = 20 * log10((sqrt(src_variance) / sqrt(noise_variance)));
    //cout << "noise mean=" << noise_mean << "  noise_variance =" << noise_variance << endl;
    return snr;
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
    //gaussian noise 10
    Mat gaussian10 = AddGaussianNoise(img_binary, 10);
    cout << "gaussian10 snr = " << SNR(img_binary, gaussian10) << endl;
    imwrite("gaussian10.bmp", gaussian10);
    //gaussian noise 30
    Mat gaussian30 = AddGaussianNoise(img_binary, 30);
    cout << "gaussian30 snr = " << SNR(img_binary, gaussian30) << endl;
    imwrite("gaussian30.bmp", gaussian30);
    //salt and pepper 0.05
    Mat salt005 = AddSaltAndPepperNoise(img_binary, 0.05);
    cout << "salt005 snr = " << SNR(img_binary, salt005) << endl;
    imwrite("salt-and-pepper005.bmp", salt005);
    //salt and pepper 0.1
    Mat salt01 = AddSaltAndPepperNoise(img_binary, 0.1);
    cout << "salt01 snr = " << SNR(img_binary, salt01) << endl;
    imwrite("salt-and-pepper01.bmp", salt01);
    //box filter 3*3
    Mat box33_gaussian10 = BoxFilter(gaussian10, 3);
    Mat box33_gaussian30 = BoxFilter(gaussian30, 3);
    Mat box33_salt005 = BoxFilter(salt005, 3);
    Mat box33_salt01 = BoxFilter(salt01, 3);
    cout << "box33_gaussian10 snr = " << SNR(img_binary, box33_gaussian10) << endl;
    cout << "box33_gaussian30 snr = " << SNR(img_binary, box33_gaussian30) << endl;
    cout << "box33_salt005 snr = " << SNR(img_binary, box33_salt005) << endl;
    cout << "box33_salt01 snr = " << SNR(img_binary, box33_salt01) << endl;
    imwrite("box33_gaussian10.bmp", box33_gaussian10);
    imwrite("box33_gaussian30.bmp", box33_gaussian30);
    imwrite("box33_salt005.bmp", box33_salt005);
    imwrite("box33_salt01.bmp", box33_salt01);
    //box filter 5*5
    Mat box55_gaussian10 = BoxFilter(gaussian10, 5);
    Mat box55_gaussian30 = BoxFilter(gaussian30, 5);
    Mat box55_salt005 = BoxFilter(salt005, 5);
    Mat box55_salt01 = BoxFilter(salt01, 5);
    cout << "box55_gaussian10 snr = " << SNR(img_binary, box55_gaussian10) << endl;
    cout << "box55_gaussian30 snr = " << SNR(img_binary, box55_gaussian30) << endl;
    cout << "box55_salt005 snr = " << SNR(img_binary, box55_salt005) << endl;
    cout << "box55_salt01 snr = " << SNR(img_binary, box55_salt01) << endl;
    imwrite("box55_gaussian10.bmp", box55_gaussian10);
    imwrite("box55_gaussian30.bmp", box55_gaussian30);
    imwrite("box55_salt005.bmp", box55_salt005);
    imwrite("box55_salt01.bmp", box55_salt01);
    //median filter 3*3
    Mat median33_gaussian10 = MedianFilter(gaussian10, 3);
    Mat median33_gaussian30 = MedianFilter(gaussian30, 3);
    Mat median33_salt005 = MedianFilter(salt005, 3);
    Mat median33_salt01 = MedianFilter(salt01, 3);
    cout << "median33_gaussian10 snr = " << SNR(img_binary, median33_gaussian10) << endl;
    cout << "median33_gaussian30 snr = " << SNR(img_binary, median33_gaussian30) << endl;
    cout << "median33_salt005 snr = " << SNR(img_binary, median33_salt005) << endl;
    cout << "median33_salt01 snr = " << SNR(img_binary, median33_salt01) << endl;
    imwrite("median33_gaussian10.bmp", median33_gaussian10);
    imwrite("median33_gaussian30.bmp", median33_gaussian30);
    imwrite("median33_salt005.bmp", median33_salt005);
    imwrite("median33_salt01.bmp", median33_salt01);
    //median filter 5*5
    Mat median55_gaussian10 = MedianFilter(gaussian10, 5);
    Mat median55_gaussian30 = MedianFilter(gaussian30, 5);
    Mat median55_salt005 = MedianFilter(salt005, 5);
    Mat median55_salt01 = MedianFilter(salt01, 5);
    cout << "median55_gaussian10 snr = " << SNR(img_binary, median55_gaussian10) << endl;
    cout << "median55_gaussian30 snr = " << SNR(img_binary, median55_gaussian30) << endl;
    cout << "median55_salt005 snr = " << SNR(img_binary, median55_salt005) << endl;
    cout << "median55_salt01 snr = " << SNR(img_binary, median55_salt01) << endl;
    imwrite("median55_gaussian10.bmp", median55_gaussian10);
    imwrite("median55_gaussian30.bmp", median55_gaussian30);
    imwrite("median55_salt005.bmp", median55_salt005);
    imwrite("median55_salt01.bmp", median55_salt01);
    //opening-then-closing
    Mat opening_then_closing_gaussian10 = Closing(Opening(gaussian10, circle_structure, 2, 2), circle_structure, 2, 2);
    Mat opening_then_closing_gaussian30 = Closing(Opening(gaussian30, circle_structure, 2, 2), circle_structure, 2, 2);
    Mat opening_then_closing_salt005 = Closing(Opening(salt005, circle_structure, 2, 2), circle_structure, 2, 2);
    Mat opening_then_closing_salt01 = Closing(Opening(salt01, circle_structure, 2, 2), circle_structure, 2, 2);
    cout << "opening_then_closing_gaussian10 snr = " << SNR(img_binary, opening_then_closing_gaussian10) << endl;
    cout << "opening_then_closing_gaussian30 snr = " << SNR(img_binary, opening_then_closing_gaussian30) << endl;
    cout << "opening_then_closing_salt005 snr = " << SNR(img_binary, opening_then_closing_salt005) << endl;
    cout << "opening_then_closing_salt01 snr = " << SNR(img_binary, opening_then_closing_salt01) << endl;
    imwrite("opening-then-closing_gaussian10.bmp", opening_then_closing_gaussian10);
    imwrite("opening-then-closing_gaussian30.bmp", opening_then_closing_gaussian30);
    imwrite("opening-then-closing_salt005.bmp", opening_then_closing_salt005);
    imwrite("opening-then-closing_salt01.bmp", opening_then_closing_salt01);
    //closing-then-opening
    Mat closing_then_opening_gaussian10 = Opening(Closing(gaussian10, circle_structure, 2, 2), circle_structure, 2, 2);
    Mat closing_then_opening_gaussian30 = Opening(Closing(gaussian30, circle_structure, 2, 2), circle_structure, 2, 2);
    Mat closing_then_opening_salt005 = Opening(Closing(salt005, circle_structure, 2, 2), circle_structure, 2, 2);
    Mat closing_then_opening_salt01 = Opening(Closing(salt01, circle_structure, 2, 2), circle_structure, 2, 2);
    cout << "closing_then_opening_gaussian10 snr = " << SNR(img_binary, closing_then_opening_gaussian10) << endl;
    cout << "closing_then_opening_gaussian30 snr = " << SNR(img_binary, closing_then_opening_gaussian30) << endl;
    cout << "closing_then_opening_salt005 snr = " << SNR(img_binary, closing_then_opening_salt005) << endl;
    cout << "closing_then_opening_salt01 snr = " << SNR(img_binary, closing_then_opening_salt01) << endl;
    imwrite("closing-then-opening_gaussian10.bmp", closing_then_opening_gaussian10);
    imwrite("closing-then-opening_gaussian30.bmp", closing_then_opening_gaussian30);
    imwrite("closing-then-opening_salt005.bmp", closing_then_opening_salt005);
    imwrite("closing-then-opening_salt01.bmp", closing_then_opening_salt01);
}