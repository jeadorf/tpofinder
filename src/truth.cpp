/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#include "tpofinder/truth.h"
#include "tpofinder/util.h"

#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace tpofinder;
using namespace std;

namespace tpofinder {

    Mat estimateHomography(const Mat& image1, const Mat& image2, const Feature& feature) {
        // Detect features
        vector<KeyPoint> kpts1, kpts2;
        Mat descs1, descs2;

        feature.detector->detect(image1, kpts1);
        feature.detector->detect(image2, kpts2);
        feature.extractor->compute(image1, kpts1, descs1);
        feature.extractor->compute(image2, kpts2, descs2);

        // Establish matches
        vector<DMatch> matches;
        feature.matcher->match(descs1, descs2, matches);

        // Find homography
        vector<Point2f> pts1, pts2;
        for (size_t i = 0; i < matches.size(); i++) {
            pts1.push_back(kpts1[matches[i].queryIdx].pt);
            pts2.push_back(kpts2[matches[i].trainIdx].pt);
        }

        if (pts1.size() < 4) {
            throw runtime_error("Cannot estimate homography: too few matches.");
        }

        Mat h = findHomography(pts1, pts2, CV_RANSAC);

        return h;
    }

    Mat HomographySequenceEstimator::next(const Mat& image) {
        if (prevImage_.empty()) {
            prevHomography_ = EYE_HOMOGRAPHY;
        } else {
            Mat u = estimateHomography(prevImage_, image, feature_);
            prevHomography_ = u * prevHomography_;
        }

        prevImage_ = image;

        // This is already the updated homography, 'previous' refers to the next
        // call to 'next'.
        return prevHomography_;
    }

}
