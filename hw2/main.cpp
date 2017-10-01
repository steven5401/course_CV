#include <iostream>
#include <climits>
#include <vector>
#include <set>
#include <utility>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
//#define DEBUG
#ifdef DEBUG
    #define D(x) cerr << x << endl
#else
    #define D(x)
#endif

int FindMinLabel(const Mat label, int i, int j, set<pair<int,int> >& equal_set) {
    int cols = label.cols, rows = label.rows;
    int top_label = INT_MAX;
    int left_label = INT_MAX;
    if (i > 0) {//top
        if (label.at<unsigned short>(i-1,j) != 0) {
            top_label = label.at<unsigned short>(i-1,j);
        }
    }
    if (j > 0) {//left
        if (label.at<unsigned short>(i,j-1) != 0) {
            left_label = label.at<unsigned short>(i,j-1);
        }
    }
    if (top_label == left_label) {
        return top_label;
    } else if (top_label > left_label) {
        if (top_label != INT_MAX) {// need sync label
            equal_set.emplace(left_label, top_label);
        }
        return left_label;
    } else {
        if (left_label != INT_MAX) {//need sync label
            equal_set.emplace(left_label, top_label);
        }
        return top_label;
    }
}

Mat FindConnectComponent(const Mat src, int& component_count, int threshold = 255) {
    component_count = 1;
    int cols = src.cols, rows = src.rows;
    Mat label(rows, cols, CV_16U, Scalar(0));
    set<pair<int,int> > equal_set;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (src.at<uchar>(i,j) >= threshold) {//need label
                int min_label = FindMinLabel(label, i, j, equal_set);
                if (min_label == INT_MAX) {//new component
                    label.at<unsigned short>(i,j) = component_count;
                    component_count++;
                } else {
                    label.at<unsigned short>(i,j) = min_label;
                }
            }
        }
    }
    if (component_count == 1) {//no component
        component_count = 0;
        return label;
    }
    int equal_table_size = component_count;
    int *equal_table = new int[equal_table_size]();//equal_table[0] is dummy
    for (int i = 0; i != equal_table_size; i++) {
        equal_table[i] = i;
    }
    for (auto p : equal_set) {
        if (equal_table[p.first] == equal_table[p.second]) continue;
        component_count--;
        int label_small = min(equal_table[p.first], equal_table[p.second]);
        int label_big = max(equal_table[p.first], equal_table[p.second]);
        for (int i = 0; i != equal_table_size; i++) {
            if (equal_table[i] == label_big) equal_table[i] = label_small;
        }
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int current_label = label.at<unsigned short>(i,j);
            if (equal_table[current_label] != current_label)
                label.at<unsigned short>(i,j) = equal_table[current_label];
        }
    }
    delete [] equal_table;
    component_count--;//because we start at 1
    return label;
}

int main(int argc, char** argv) {
    Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    int histogram_array[256] = {};
    if (!img_gray.empty()) {
        int cols = img_gray.cols, rows = img_gray.rows;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                histogram_array[img_gray.at<uchar>(i,j)] += 1;
                if (img_gray.at<uchar>(i,j) >= 128) {
                    img_gray.at<uchar>(i,j) = 255;
                } else {
                    img_gray.at<uchar>(i,j) = 0;
                }
            }
        }
        imwrite("binary_lena.bmp", img_gray);
        Mat histogram_img(256,256,CV_8UC1,Scalar(255));
        float scale = 0.95 * 256 / (*max_element(histogram_array, histogram_array + 256));
        for (int i = 0; i < 256; i++) {
            int intensity = (int)(scale * histogram_array[i]);
            line(histogram_img, Point(i,255), Point(i,255-intensity), Scalar(0));
        }
        imwrite("histogram.bmp", histogram_img);
        int component_count = 0;
        Mat label = FindConnectComponent(img_gray, component_count, 1);
        int max_label = 0;
        rows = label.rows;
        cols = label.cols;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (label.at<unsigned short>(i,j) > max_label) {
                    max_label = label.at<unsigned short>(i,j);
                }
            }
        }
        D("max label=" << max_label);
        vector<Point> *component_set = new vector<Point>[max_label + 1];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (label.at<unsigned short>(i,j) != 0) {
                    component_set[label.at<unsigned short>(i,j)].push_back(Point(j,i));//due to image coordinate
                }
            }
        }
        Mat components_img;
        cvtColor(img_gray, components_img, COLOR_GRAY2BGR);
        for (int i = 0; i != max_label + 1; i++) {//draw each bounding box
            if (component_set[i].size() >= 500) {//ignore patch that is small than 500 pixels
                Rect r = boundingRect(component_set[i]);
                rectangle(components_img, r.tl(), r.br() - Point(1,1), Scalar(255,0,0), 2, 8, 0);
                Point center = (r.tl() + r.br()) / 2;
                //line(components_img, Point(center.x - 5, center.y), Point(center.x + 5, center.y), Scalar(255,0,0));
                //line(components_img, Point(center.x, center.y - 5), Point(center.x, center.y + 5), Scalar(255,0,0));
            }
        }
        imwrite("components.bmp", components_img);
    } else {
        if (argc == 2) {
            cout << "no such file:" << argv[1] << endl;
        } else {
            cout << "command line argument is missed" << endl;
        }
    }
}