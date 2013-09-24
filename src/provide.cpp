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

    void WebcamImageProvider::capture_loop() {
        cv::Mat buffer;
        while (!stop_) {
            capture_ >> buffer;
            image_mutex_.lock();
            image_ = buffer;
            ready_.notify_one();
            image_mutex_.unlock();
        }
    }
    
    WebcamImageProvider::WebcamImageProvider() : capture_(0), image_mutex_(),
            image_(), ready_(), stop_(false) {
        if (!capture_.isOpened()) {
            cerr << "Could not open default camera." << endl;
            exit(-1);
        }
        worker_ = new thread(&WebcamImageProvider::capture_loop, std::ref(*this));
    }

    WebcamImageProvider::~WebcamImageProvider() {
        stop_ = true;
        worker_->join();
        delete worker_;
    }

    bool WebcamImageProvider::next(cv::Mat &image) {
        std::unique_lock<std::mutex> lock(image_mutex_);
        ready_.wait(lock);
        image = image_;
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
