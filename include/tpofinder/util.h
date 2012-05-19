/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#ifndef UTIL_H
#define	UTIL_H

#include <boost/filesystem.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace tpofinder {

    cv::Mat readHomography(const boost::filesystem::path& path);

    void writeHomography(const boost::filesystem::path& path,
            const cv::Mat& homography);

    void invertHomography(const boost::filesystem::path& src, const boost::filesystem::path& dst);
    
    cv::Scalar readColor(const boost::filesystem::path& path);

    void writeColor(const boost::filesystem::path& path, const cv::Scalar& color);
    
    void perspectiveTransformKeypoints(const std::vector<cv::KeyPoint>& src,
            std::vector<cv::KeyPoint>& dst,
            const cv::Mat& mtx);

    std::vector<int> findInliers(const std::vector<cv::Point2f>& pts1,
            const std::vector<cv::Point2f>& pts2, const cv::Mat& homography,
            const float reprojThreshold = 3.0);
    
}

#endif
