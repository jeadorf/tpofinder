/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#include "tpofinder/detect.h"
#include "tpofinder/util.h"

#include <boost/foreach.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <stdarg.h>

using namespace cv;
using namespace std;

namespace tpofinder {

    Detector::Detector(const Modelbase& modelbase, const Feature& feature,
            const cv::Ptr<DetectionFilter> filter, double reprojThreshold) :
    /*       */ modelbase_(modelbase), feature_(feature), filter_(filter),
    /*       */ reprojThreshold_(reprojThreshold) {
        vector<Mat> descriptors;

        BOOST_FOREACH(const PlanarModel& m, modelbase_.models) {
            descriptors.push_back(m.allDescriptors);
        }
        feature_.matcher->add(descriptors);
        feature_.matcher->train();
    }

    Scene Detector::describe(const Mat& sceneImage) {
        CV_Assert(!sceneImage.empty());
        vector<KeyPoint> kpts;
        feature_.detector->detect(sceneImage, kpts);
        cv::Mat descs;
        feature_.extractor->compute(sceneImage, kpts, descs);
        return Scene(sceneImage, kpts, descs);
    }

    vector<DMatch> Detector::match(const Scene& scene) {
        vector<DMatch> matches;
        feature_.matcher->match(scene.descriptors, matches);
        return matches;
    }

    vector<Detection> Detector::detect(const Scene& scene) {
        vector<Detection> detections;
        vector<DMatch> matches = match(scene);

        for (size_t i = 0; i < modelbase_.models.size(); i++) {
            const PlanarModel& model = modelbase_.models[i];

            vector<Point2f> scenePoints, modelPoints;
            vector<DMatch> modelMatches;
            for (size_t j = 0; j < matches.size(); j++) {
                if ((size_t) matches[j].imgIdx == i) {
                    scenePoints.push_back(scene.keypoints[matches[j].queryIdx].pt);
                    modelPoints.push_back(model.allKeypoints[matches[j].trainIdx].pt);
                    modelMatches.push_back(matches[j]);
                }
            }

            if (scenePoints.size() >= 4) {
                // TODO: Depending on whether they use symmetric error criteria
                // for determining RANSAC inliers, it might be a difference
                // whether the homography between model and scene is computed
                // or its inverse homography (i.e. between scene and model).
                Mat h = findHomography(modelPoints, scenePoints, CV_RANSAC, reprojThreshold_);
                vector<int> inliers = findInliers(modelPoints, scenePoints, h, reprojThreshold_);
                Detection d(model, h, modelMatches, inliers);
                if (filter_->accept(d)) {
                    detections.push_back(d);
                }
            }
        }

        return detections;
    }

    bool MagicHomographyFilter::accept(const Detection& detection) {
        Mat h = detection.homography;
        double sx = h.at<double>(0, 0);
        double sy = h.at<double>(1, 1);
        double kx = h.at<double>(0, 1);
        double ky = h.at<double>(1, 0);

        double c = 10.0 * (max(sx, sy) / min(sx, sy)) + 5.0 * kx + 5.0 * ky + 1.0 / min(sx, sy);
        return c > 10;
    }

    bool InliersRatioFilter::accept(const Detection& detection) {
        if (detection.matches.size() == 0) {
            return false;
        } else {
            return detection.inliers.size() / (double) detection.matches.size() >= threshold;
        }
    }

    bool EigenvalueFilter::accept(const Detection& detection) {
        Mat linpart = detection.homography(Range(0, 2), Range(0, 2));
        Mat eigenvals;
        eigen(linpart, eigenvals);
        CV_Assert(eigenvals.type() == CV_64FC1);
        double v1 = eigenvals.at<double>(0);
        double v2 = eigenvals.at<double>(1);
        if (v1 < minEigenvalue || v1 > maxEigenvalue) {
            return false;
        }
        if (v2 < minEigenvalue || v2 > maxEigenvalue) {
            return false;
        }
        return true;
    }

    bool AndFilter::accept(const Detection& detection) {
        return lfilter_->accept(detection) && rfilter_->accept(detection);
    }

}
