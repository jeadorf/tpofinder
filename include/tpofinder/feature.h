/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#ifndef FEATURE_H
#define	FEATURE_H

#include <opencv2/features2d/features2d.hpp>

namespace tpofinder {

    struct Feature {
        Feature(const std::string& detectorName = "ORB",
                const std::string& extractorName = "ORB",
                const std::string& matcherName = "BruteForce-Hamming");

        Feature(const cv::Ptr<cv::FeatureDetector> detector,
                const cv::Ptr<cv::DescriptorExtractor> extractor,
                const cv::Ptr<cv::DescriptorMatcher> matcher);

        cv::Ptr<cv::FeatureDetector> detector;
        cv::Ptr<cv::DescriptorExtractor> extractor;
        cv::Ptr<cv::DescriptorMatcher> matcher;
        
    private:
        
        void validate();

    };

}

#endif
