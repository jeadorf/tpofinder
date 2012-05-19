/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#ifndef MODEL_H
#define	MODEL_H

#include "tpofinder/core.h"
#include "tpofinder/feature.h"

#include <boost/filesystem.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

namespace tpofinder {

    struct PlanarView {

        PlanarView() {
            /* constructed object is invalid */
        }

        PlanarView(const cv::Mat& image, const cv::Mat& roi,
                const cv::Mat& homography,
                const std::vector<cv::KeyPoint>& keypoints,
                const cv::Mat & descriptors) :
        /*       */ image(image), roi(roi), homography(homography),
        /*       */ keypoints(keypoints), descriptors(descriptors) {
            /* no operation */
        }

        /** Image of the object (CV_8UC3, BGR). */
        cv::Mat image;
        /** Region of interest (CV_8UC1, binary). */
        cv::Mat roi;
        /** Maps keypoints from the reference view onto this view. */
        cv::Mat homography;
        /** Keypoints relative to this view. */
        std::vector<cv::KeyPoint> keypoints;
        /** Descriptors associated with this view. */
        cv::Mat descriptors;

        static PlanarView create(const cv::Mat& image, const cv::Mat& roi,
                const cv::Mat& homography = EYE_HOMOGRAPHY,
                const Feature& feature = Feature());

        static PlanarView load(const boost::filesystem::path& path,
                const cv::Mat& referenceRoi,
                const Feature& feature = Feature());

    };

    struct PlanarModel {

        PlanarModel() {
            /* constructed object is invalid */
        }

        PlanarModel(const std::string& name, const cv::Scalar& color,
                const std::vector<PlanarView>& views);

        /** Identifies this object. */
        std::string name;
        /** Used for visualization purposes. */
        cv::Scalar color;
        /** All views together make up this planar model. The first view is
         * designated reference; its homography has to equal the identity
         * matrix. */
        std::vector<PlanarView> views;
        /** Collection of all keypoints from all views transformed into the
         * reference frame of the first view. Be careful, some duplication here. */
        std::vector<cv::KeyPoint> allKeypoints;
        /** Collection of all descriptors. Be careful, some duplication here. */
        cv::Mat allDescriptors;

        static PlanarModel create(const std::string& name,
                const cv::Mat& image, const cv::Mat& roi,
                const cv::Scalar& color = cv::Scalar(0, 0, 255, 255),
                const Feature& feature = Feature());

        static PlanarModel load(const boost::filesystem::path& path,
                const Feature& feature = Feature());

    };

    class Modelbase {
    public:

        Modelbase(const Feature& feature = Feature()) : feature_(feature) {}

        void add(const PlanarModel & model) {
            models.push_back(model);
        }

        /** Equivalent to Modelbase::add(PlanarModel::load( ... )). */
        void add(const boost::filesystem::path& path) {
            add(PlanarModel::load(path, feature_));
        }

        int findByName(const std::string& name);
        
        std::vector<PlanarModel> models;
        
    private:
        Feature feature_;
    };

}

#endif
