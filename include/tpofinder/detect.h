/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#ifndef DETECT_H
#define	DETECT_H

#include "tpofinder/model.h"

#include <numeric>
#include <opencv2/features2d/features2d.hpp>

namespace tpofinder {

    /** Describes the scene in which objects are to be detected. */
    struct Scene {

        Scene() {
            /* constructs an invalid object */
        }

        Scene(const cv::Mat& image,
                const std::vector<cv::KeyPoint>& keypoints,
                const cv::Mat & descriptors) :
        /*       */ image(image), keypoints(keypoints),
        /*       */ descriptors(descriptors) {
            /* no operation */
        }

        /** Image of the query scene. */
        cv::Mat image;
        /** All keypoints of the query scene. */
        std::vector<cv::KeyPoint> keypoints;
        /** All descriptors of the query scene. */
        cv::Mat descriptors;

    };

    /** A description of objects that are presumably on scene. */
    struct Detection {

        Detection() {
            /* this constructor is required for testing */
        }

        Detection(const PlanarModel& model, const cv::Mat& homography,
                const std::vector<cv::DMatch>& matches,
                const std::vector<int>& inliers) :
        /*       */ model(model), homography(homography),
        /*       */ matches(matches), inliers(inliers) {
            /* no operation */
        }

        /** Corresponds to the detected object. */
        PlanarModel model;
        /** Transforms model coordinates into scene coordinates. */
        cv::Mat homography;
        /** Matches that lead to this detection. For a given DMatch dm,
         * dm.queryIdx is an index for the keypoints (descriptors) of the scene;
         * dm.trainIdx references the keypoints (descriptors) of the model as
         * stored in PlanarModel::allKeyppoints (PlanarModel::allDescriptors);
         * dm.imgIdx references the planar model it belongs to. The planar view
         * the matched keypoints belongs to can be recovered by counting the
         * numbers of keypoints in each view. */
        std::vector<cv::DMatch> matches;
        /** References those matches that are considered inliers for the given
         * fitted homography. */
        std::vector<int> inliers;

    };

    struct DetectionFilter {

        virtual ~DetectionFilter() {
            // Always declare the constructor of the base class virtual when dealing
            // with inheritance.
        }

        virtual bool accept(const Detection & detection) = 0;

    };

    struct AcceptAllFilter : public DetectionFilter {

        virtual bool accept(const Detection & detection) {
            return true;
        }
    };

    struct MagicHomographyFilter : public DetectionFilter {
        virtual bool accept(const Detection & detection);
    };

    struct InliersRatioFilter : public DetectionFilter {

        InliersRatioFilter(float threshold = 0.10) : threshold(threshold) {
        }

        virtual bool accept(const Detection & detection);

        float threshold;

    };

    struct EigenvalueFilter : public DetectionFilter {

        EigenvalueFilter(double minEigenvalue = std::numeric_limits<double>().min(),
                double maxEigenvalue = std::numeric_limits<double>().max()) :
        minEigenvalue(minEigenvalue), maxEigenvalue(maxEigenvalue) {
        }

        virtual bool accept(const Detection & detection);

        double minEigenvalue;
        double maxEigenvalue;
    };

    class AndFilter : public DetectionFilter {
    public:

        AndFilter(const cv::Ptr<DetectionFilter> lfilter,
                const cv::Ptr<DetectionFilter> rfilter)
        /*    */ : lfilter_(lfilter), rfilter_(rfilter) {
        }

        virtual bool accept(const Detection & detection);

    private:
        cv::Ptr<DetectionFilter> lfilter_;
        cv::Ptr<DetectionFilter> rfilter_;

    };

    /** Detects objects in a scene. */
    class Detector {
    public:

        Detector(const Modelbase& modelbase = Modelbase(),
                const Feature& feature = Feature(),
                const cv::Ptr<DetectionFilter> filter = new AcceptAllFilter(),
                double reprojThreshold = 3.0);

        /** Construct a scene description out of an image. */
        Scene describe(const cv::Mat& sceneImage);

        /** Detect objects given the description of a scene. */
        std::vector<Detection> detect(const Scene& scene);

        const Modelbase& modelbase() const {
            return modelbase_;
        }

    private:

        std::vector<cv::DMatch> match(const Scene& scene);

        Modelbase modelbase_;
        Feature feature_;
        cv::Ptr<DetectionFilter> filter_;
        float reprojThreshold_;

    };

}

#endif
