 #include "tpofinder/core.h"
#include "tpofinder/feature.h"
#include "tpofinder/util.h"
#include "tpofinder/visualize.h"

#include <cstdio>
#include <gtest/gtest.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace tpofinder;
namespace bfs = boost::filesystem;

class util : public ::testing::Test {
public:

    virtual void SetUp() {
        // This is the homography computed from two quadrilaterals marked on
        // true separate views of the same object.
        pts1 = (Mat_<Point2f > (4, 1) <<
                Point2f(207, 112),
                Point2f(226, 319),
                Point2f(416, 365),
                Point2f(466, 203));
        pts2 = (Mat_<Point2f > (4, 1) <<
                Point2f(116, 80),
                Point2f(146, 253),
                Point2f(262, 334),
                Point2f(275, 184));
        homography = findHomography(pts1, pts2);

        srand(42);
        nOutliers = 200;
        for (size_t i = 0; i < 1000; i++) {
            iPts1.push_back(Point2f(rand() * 100, rand() * 100));
        }
        for (size_t i = 0; i < iPts1.size() - nOutliers; i++) {
            iPts2.push_back(iPts1[i]);
        }
        for (size_t i = iPts1.size() - nOutliers; i < iPts1.size(); i++) {
            iPts2.push_back(Point2f(101 + rand() * 100, 101 + rand() * 100));
        }
    }

    void expectNoDuplicates(const vector<int>& inliers) {
        for (size_t i = 0; i < inliers.size(); i++) {
            for (size_t j = i + 1; j < inliers.size(); j++) {
                EXPECT_NE(inliers[i], inliers[j]);
            }
        }
    }

    void expectIndicesWithinBounds(const vector<int>& inliers, int imin, int imax) {
        for (size_t i = 0; i < inliers.size(); i++) {
            EXPECT_GE(inliers[i], imin);
            EXPECT_LE(inliers[i], imax);
        }
    }

    // Data for homography tests
    Mat_<Point2f> pts1;
    Mat_<Point2f> pts2;
    Mat homography;

    // Data for findInliers tests
    vector<Point2f> iPts1;
    vector<Point2f> iPts2;
    size_t nOutliers;
};

TEST_F(util, findHomographyReturnsDouble) {
    EXPECT_EQ(homography.type(), CV_64FC1);
}

TEST_F(util, writeReadHomography) {
    char *tmp = tmpnam(NULL);
    bfs::path p = string(tmp);
    writeHomography(p, homography);
    Mat h = readHomography(p);
    EXPECT_EQ(h.type(), CV_64FC1);
    EXPECT_NEAR(norm(homography - h), 0, 0.001);
    bfs::remove(p);
}

TEST_F(util, invertHomography) {
    char *tmp = tmpnam(NULL);
    bfs::path p = "data/blokus/002.yml";
    bfs::path q = string(tmp);
    Mat ref = imread("data/blokus/ref.jpg");
    Mat image002 = imread("data/blokus/002.jpg");
    Mat g = readHomography(p);
    Mat out = blend(ref, image002, g);
    imshow("util.invertHomography.normal", out);
    invertHomography(p, q);
    Mat h = readHomography(q);
    out = blend(image002, ref, h);
    imshow("util.invertHomography.inverted", out);
    bfs::remove(q);
}

TEST_F(util, writeReadColor) {
    char *tmp = tmpnam(NULL);
    bfs::path p = string(tmp);
    Scalar c(0, 255, 0, 123);
    writeColor(p, c);
    Scalar d = readColor(p);
    EXPECT_NEAR(norm(c - d), 0, 0.001);
    bfs::remove(p);
}

TEST_F(util, perspectiveTransformKeypointsBackAndForth) {
    Mat image = imread("data/blokus/ref.jpg");
    vector<KeyPoint> kpts;
    Feature f;
    f.detector->detect(image, kpts);
    vector<KeyPoint> dst, kpts_;
    perspectiveTransformKeypoints(kpts, dst, homography);
    Mat invHomography;
    invert(homography, invHomography);
    perspectiveTransformKeypoints(dst, kpts_, invHomography);
    for (size_t i = 0; i < kpts.size(); i++) {
        EXPECT_NEAR(norm(kpts[i].pt - kpts_[i].pt), 0, 0.0001);
    }
}

/** When model and scene points are equal and the homography is the identity,
 * then all the matches are considered inliers. */
TEST_F(util, identityInliersCount) {
    vector<int> inliers = findInliers(iPts1, iPts1, EYE_HOMOGRAPHY);
    EXPECT_EQ(inliers.size(), iPts1.size());
}

TEST_F(util, identityInliersNoDuplicates) {
    vector<int> inliers = findInliers(iPts1, iPts1, EYE_HOMOGRAPHY);
    expectNoDuplicates(inliers);
}

TEST_F(util, identityInliersIndicesWithinBounds) {
    vector<int> inliers = findInliers(iPts1, iPts1, EYE_HOMOGRAPHY);
    expectIndicesWithinBounds(inliers, 0, iPts1.size()-1);
}

/** When model and scene points are equal and the homography represents a large
 * scale translation, then there are no inliers which support this homography.
 */
TEST_F(util, largeTranslationNoInliers) {
    Mat translate = (Mat_<double>(3, 3) << 1, 0, 110, 0, 1, 110, 0, 0, 1);
    vector<int> inliers = findInliers(iPts1, iPts1, translate);
    EXPECT_EQ(inliers.size(), 0);
}

/** When model and scene points are equal except for n points and the homography
 * is the identity, then we have n outliers. */
TEST_F(util, identityPlusNoiseInliersCount) {
    vector<int> inliers = findInliers(iPts1, iPts2, EYE_HOMOGRAPHY);
    EXPECT_EQ(inliers.size(), iPts1.size() - nOutliers);
}

TEST_F(util, identityPlusNoiseInliersNoDuplicates) {
    vector<int> inliers = findInliers(iPts1, iPts2, EYE_HOMOGRAPHY);
    expectNoDuplicates(inliers);
}

TEST_F(util, identityPlusNoiseInliersIndicesWithinBounds) {
    vector<int> inliers = findInliers(iPts1, iPts2, EYE_HOMOGRAPHY);
    expectIndicesWithinBounds(inliers, 0, iPts1.size() - 1);
}

TEST_F(util, identityPlusNoiseInliersSymmetric) {
    // Check for symmetry as well
    vector<int> inliers = findInliers(iPts1, iPts2, EYE_HOMOGRAPHY);
    vector<int> inliers2 = findInliers(iPts2, iPts1, EYE_HOMOGRAPHY);
    EXPECT_EQ(inliers.size(), inliers2.size());
}
