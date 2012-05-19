/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#ifndef CORE_H
#define	CORE_H

#include <opencv2/core/core.hpp>

namespace tpofinder {

    const cv::Mat EYE_HOMOGRAPHY = cv::Mat::eye(3, 3, CV_64FC1);

}

#endif
