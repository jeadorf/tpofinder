/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#include "tpofinder/provide.h"

#include <iostream>

using namespace cv;
using namespace std;

namespace tpofinder {

    WebcamImageProvider::WebcamImageProvider() : capture_(new VideoCapture(0)) {
        if (!capture_->isOpened()) {
            cerr << "Could not open default camera." << endl;
            exit(-1);
        }
    }

    WebcamImageProvider::~WebcamImageProvider() {
        delete capture_;
    }

    bool WebcamImageProvider::next(Mat &image) {
        *capture_ >> image;
        return true;
    }

    
    bool StdinFilenameImageProvider::next(Mat &image) {
        string s;
        getline(cin, s);
        if (s.empty()) {
            return false;
        } else {
            image = imread(s);
        }
        if (image.empty()) {
            return false;
        } else {
            return true;
        }
    }

    ListFilenameImageProvider::ListFilenameImageProvider(const vector<string>
            &files) : num_(0), files_(files) {}

    bool ListFilenameImageProvider::next(Mat &image) {
        if (num_ < files_.size()) {
            string f = files_[num_++];
            image = imread(f);
        } else {
            return false;
        }
    }

}
