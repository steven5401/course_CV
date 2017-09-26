#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main(int argc, char** argv) {
    Mat src = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    if (!src.empty()) {
        Mat upside_down = src.clone();
        int cols = upside_down.cols, rows = upside_down.rows;
        for (int i = 0; i < rows / 2; i++) {
            for (int j = 0; j < cols; j++)
                swap(upside_down.at<Vec3b>(i,j), upside_down.at<Vec3b>(rows - i, j));
        }
        imwrite("upside_down.bmp", upside_down);
        Mat right_side_left = src.clone();
        for (int j = 0; j < cols / 2; j++) {
            for (int i = 0; i < rows; i++)
                swap(right_side_left.at<Vec3b>(i,j), right_side_left.at<Vec3b>(i, cols - j));
        }
        imwrite("right_side_left.bmp", right_side_left);
        Mat diagonal_mirror = src.clone();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++)
                diagonal_mirror.at<Vec3b>(i,j) = src.at<Vec3b>(j, i);
        }
        imwrite("diagonal_mirror.bmp", diagonal_mirror);
    } else {
        cout << "no such file:" << argv[1] << endl;
    }
}