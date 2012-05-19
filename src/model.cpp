/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#include "tpofinder/model.h"
#include "tpofinder/util.h"

#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace tpofinder;
using namespace std;
namespace bfs = boost::filesystem;

namespace tpofinder {

    PlanarView PlanarView::create(const Mat& image, const Mat& roi,
            const Mat& homography, const Feature& feature) {
        CV_Assert(!image.empty());
        CV_Assert(!roi.empty());
        CV_Assert(roi.channels() == 1);

        vector<KeyPoint> keypoints;
        // TODO: convert to grayscale (SIFT, SURF ...)
        feature.detector->detect(image, keypoints, roi);
        Mat descriptors;
        feature.extractor->compute(image, keypoints, descriptors);

        return PlanarView(image, roi > 0, homography, keypoints, descriptors);
    }

    PlanarView PlanarView::load(const bfs::path& path,
            const Mat& referenceRoi, const Feature& feature) {
        Mat h = readHomography(path);
        bfs::path imgPath = path;
        imgPath.replace_extension(".jpg");
        Mat image = imread(imgPath.string());
        CV_Assert(!image.empty());

        // Map the region of interest from the reference image to the
        // training image.
        Mat roi;
        warpPerspective(referenceRoi, roi, h, image.size());

        return PlanarView::create(image, roi, h, feature);
    }

    PlanarModel::PlanarModel(const string& name, const Scalar& color,
            const vector<PlanarView>& views) : name(name), color(color), views(views) {

        BOOST_FOREACH(const PlanarView& v, views) {
            vector<KeyPoint> kptsInRef;
            Mat hInv;
            invert(v.homography, hInv);
            perspectiveTransformKeypoints(v.keypoints, kptsInRef, hInv);
            for (size_t i = 0; i < kptsInRef.size(); i++) {
                allKeypoints.push_back(kptsInRef[i]);
                allDescriptors.push_back(v.descriptors.row(i));
            }
        }
    }

    PlanarModel PlanarModel::create(const std::string& name,
            const Mat& image, const Mat& roi,
            const Scalar& color, const Feature& feature) {
        CV_Assert(!image.empty());
        CV_Assert(!roi.empty());

        vector<PlanarView> views;
        views.push_back(PlanarView::create(image, roi, EYE_HOMOGRAPHY, feature));
        return PlanarModel(name, color, views);
    }

    PlanarModel PlanarModel::load(const bfs::path& path, const Feature& feature) {
        CV_Assert(bfs::exists(path / "ref.jpg"));
        CV_Assert(bfs::exists(path / "roi.png"));
        CV_Assert(bfs::exists(path / "info.yml"));

        Mat ref = imread((path / "ref.jpg").string());
        Mat roi = imread((path / "roi.png").string(), 0);
        Scalar color = readColor(path / "info.yml");

        // Ensure that the alpha channel is set approximately to 1.0 such that
        // the color is not transparent. While OpenCV does not use this alpha
        // channel by default, other applications do.
        CV_Assert(color[3] == 255);
        CV_Assert(!ref.empty());
        CV_Assert(!roi.empty());

        vector<PlanarView> views;
        views.push_back(PlanarView::create(ref, roi, EYE_HOMOGRAPHY, feature));

        // The views must be loaded in the correct order; the files must follow
        // a certain naming scheme. This ensures reproducibility.
        bfs::path p(path / "001.yml");
        int i = 2;
        while (bfs::exists(p)) {
            views.push_back(PlanarView::load(p, views[0].roi, feature));
            p = path / str(boost::format("%03d.yml") % i);
            i++;
        }

        return PlanarModel(path.leaf().string(), color, views);
    }

    int Modelbase::findByName(const string& name) {
        for (size_t i = 0; i < models.size(); i++) {
            if (models[i].name == name) {
                return i;
            }
        }
        return -1;
    }

}
