/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#include "tpofinder/core.h"
#include "tpofinder/feature.h"

#include <opencv2/core/core.hpp>
#include <vector>

#ifndef TRUTH_H
#define	TRUTH_H

namespace tpofinder {

    cv::Mat estimateHomography(const cv::Mat& image1, const cv::Mat& image2,
            const Feature& feature = Feature());

    class HomographySequenceEstimator {
    public:

        HomographySequenceEstimator(const Feature& feature = Feature()) :
        /*       */ feature_(feature) {
            /* do nothing */
        }

        cv::Mat next(const cv::Mat& image);

    private:
        Feature feature_;
        cv::Mat prevHomography_;
        cv::Mat prevImage_;
        // We could also preserve the keypoints and descriptors of the previous
        // image such that we do not need to recompute them. Currently, since
        // this estimator is part of offline training, there is no need for such
        // a speed-up.

        //        cv::Mat prevDescs_;
        //        cv::Mat prevKpts_;
    };

}

#endif
