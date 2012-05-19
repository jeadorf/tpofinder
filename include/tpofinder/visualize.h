/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#ifndef VISUALIZE_H
#define	VISUALIZE_H

#include "tpofinder/detect.h"

#include "opencv2/highgui/highgui.hpp"

/** Visualizes models, scenes, and detections. The drawing routines that return
 * an image (cv::Mat), always return a cloned image that does not share data.
 * Drawing routines that take a parameter for storing return values just draw
 * over the out image. */

namespace tpofinder {

    cv::Mat drawModel(const PlanarModel& model);

    cv::Mat drawScene(const Scene& scene, bool keypoints = true);

    void drawMatches(cv::Mat& out, const Scene& scene, const Detection& detection);

    void drawModelContour(cv::Mat& out, const PlanarModel& model,
            const cv::Mat& homography, const std::string& label = std::string());

    void drawDetection(cv::Mat& out, const Detection& detection);

    void drawCenteredText(cv::Mat& out, const std::string& text, const cv::Point& org,
            const cv::Scalar& color = cv::Scalar::all(0), int thickness = 1,
            double fontScale = 1, int fontFace = CV_FONT_HERSHEY_SIMPLEX);

    cv::Mat blend(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& homography);

}

#endif
