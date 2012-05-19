/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#include <boost/foreach.hpp>

#include "tpofinder/util.h"

using namespace cv;
namespace bfs = boost::filesystem;

namespace tpofinder {

    Mat readHomography(const bfs::path& path) {
        FileStorage in(path.string(), FileStorage::READ);
        Mat h;
        in["homography"] >> h;
        in.release();
        return h;
    }

    void writeHomography(const bfs::path& path, const Mat& homography) {
        FileStorage out(path.string(), FileStorage::WRITE);
        out << "homography" << homography;
        out.release();
    }

    void invertHomography(const bfs::path& src, const bfs::path& dst) {
        Mat h = readHomography(src);
        Mat hInv;
        invert(h, hInv);
        hInv /= hInv.at<double>(2, 2);
        writeHomography(dst, hInv);
    }

    Scalar readColor(const bfs::path& path) {
        FileStorage in(path.string(), FileStorage::READ);
        Mat c;
        in["color"] >> c;
        in.release();
        return Scalar(c.at<uint8_t > (0, 0), c.at<uint8_t > (0, 1), c.at<uint8_t > (0, 2), c.at<uint8_t >(0, 3));
    }

    void writeColor(const bfs::path& path, const Scalar& color) {
        FileStorage out(path.string(), FileStorage::WRITE);
        Mat c = (Mat_<uint8_t > (1, 4) << color[0], color[1], color[2], color[3]);
        out << "color" << c;
        out.release();
    }

    void perspectiveTransformKeypoints(const vector<KeyPoint>& src,
            vector<KeyPoint>& dst, const Mat& mtx) {
        Mat_<Point2f> srcMat;
        for (size_t i = 0; i < src.size(); i++) {
            srcMat.push_back(src[i].pt);
        }
        Mat_<Point2f> dstMat;
        perspectiveTransform(srcMat, dstMat, mtx);
        for (size_t i = 0; i < src.size(); i++) {
            KeyPoint k = src[i];
            k.pt = dstMat.at<Point2f > (i);
            dst.push_back(k);
        }
    }

    vector<int> findInliers(const vector<Point2f>& pts1, const vector<Point2f>& pts2,
            const Mat& homography, const float reprojThreshold) {
        CV_Assert(pts1.size() == pts2.size());
        CV_Assert(!homography.empty());

        vector<int> inliers;
        vector<Point2f> pts1_;
        perspectiveTransform(pts1, pts1_, homography);
        for (size_t i = 0; i < pts1.size(); i++) {
            if (norm(pts1_[i] - pts2[i]) <= reprojThreshold) {
                inliers.push_back(i);
            }
        }
        return inliers;
    }

}
