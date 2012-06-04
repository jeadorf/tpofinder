#include "tpofinder/configure.h"
#include "tpofinder/truth.h"
#include "tpofinder/visualize.h"

#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

using namespace cv;
using namespace tpofinder;
using namespace std;

class truth : public ::testing::Test {
public:

    void SetUp() {

    }
    
    HomographySequenceEstimator estimator;
    
};

TEST_F(truth, estimateIdentity) {
    Mat image = imread(PROJECT_BINARY_DIR + "/data/test/scene-blokus-taco-1.png");
    Mat homography = estimator.next(image);
    EXPECT_NEAR(norm(homography - EYE_HOMOGRAPHY), 0, 0.2);
}

TEST_F(truth, estimateOneFrame) {
    Mat image1 = imread(PROJECT_BINARY_DIR + "/data/test/scene-blokus-taco-1.png");
    Mat image2 = imread(PROJECT_BINARY_DIR + "/data/test/scene-blokus-taco-2.png");
    
    Mat homography1 = estimator.next(image1);
    Mat homography2 = estimator.next(image2);
    
    Mat out = blend(image1, image2, homography2);
    imshow("truth.estimateOneFrame", out);
}
