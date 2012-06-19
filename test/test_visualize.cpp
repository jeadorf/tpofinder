#include "test.h"
#include "tpofinder/configure.h"
#include "tpofinder/detect.h"
#include "tpofinder/util.h"
#include "tpofinder/visualize.h"

#include <boost/format.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace tpofinder;

class visualize : public ::testing::Test {
public:

    virtual void SetUp() {
        Modelbase modelbase;
        //        modelbase.add("data/blokus");
        modelbase.add(PROJECT_BINARY_DIR + "/data/taco");
        detector = Detector(modelbase);
        Mat image = imread(PROJECT_BINARY_DIR + "/data/test/scene-blokus-taco-1.png");
        ASSERT_FALSE(image.empty());
        scene = detector.describe(image);
    }

    Scene scene;
    Detector detector;

};

TEST_F(visualize, drawCenteredText) {
    Mat out = 255 * Mat::ones(480, 640, CV_8UC1);
    drawCenteredText(out, "centered", Point(320, 240), Scalar::all(128), 2);
    imshow("visualize.putTextCentered", out);
}

TEST_F(visualize, drawModel) {
    Mat out = drawModel(detector.modelbase().models[0]);
    imshow("visualize.drawModel", out);
}

TEST_F(visualize, drawScene) {
    Mat out = drawScene(scene);
    imshow("visualize.drawScene", out);
}

TEST_F(visualize, cvDrawMatches) {
    Mat out;
    Scalar white = CV_RGB(255, 255, 255);
    Scalar gray = Scalar::all(128);
    Mat sceneImage = Mat::zeros(480, 640, CV_8UC3);
    drawCenteredText(sceneImage, "scene", Point(320, 240), white);
    Mat modelImage;
    multiply(gray, Mat::ones(480, 640, CV_8UC3), modelImage);
    drawCenteredText(modelImage, "model", Point(320, 240), white);

    vector<KeyPoint> sceneKeypoints;
    sceneKeypoints.push_back(KeyPoint(30, 60, 1));
    sceneKeypoints.push_back(KeyPoint(40, 150, 1));
    sceneKeypoints.push_back(KeyPoint(460, 400, 1));
    for (size_t i = 0; i < sceneKeypoints.size(); i++) {
        string n = str(boost::format(" %d") % i);
        drawCenteredText(sceneImage, n, sceneKeypoints[i].pt, white);
    }
    drawKeypoints(sceneImage, sceneKeypoints, sceneImage);

    vector<KeyPoint> modelKeypoints;
    modelKeypoints.push_back(KeyPoint(550, 60, 1));
    modelKeypoints.push_back(KeyPoint(50, 230, 1));
    modelKeypoints.push_back(KeyPoint(320, 420, 1));
    for (size_t i = 0; i < modelKeypoints.size(); i++) {
        string n = str(boost::format(" %d") % i);
        drawCenteredText(modelImage, n, modelKeypoints[i].pt, white);
    }
    drawKeypoints(modelImage, modelKeypoints, modelImage);

    vector<DMatch> matches;
    matches.push_back(DMatch(0, 1, 0.1));
    matches.push_back(DMatch(1, 2, 0.3));
    matches.push_back(DMatch(2, 0, 0.2));

    cv::drawMatches(sceneImage, sceneKeypoints, modelImage, modelKeypoints, matches, out);
    imshow("visualize.cvDrawMatches", out);
}

TEST_F(visualize, drawMatches) {
    vector<Detection> detections = detector.detect(scene);
    ASSERT_GE(detections.size(), 1);
    Mat out = drawScene(scene);
    drawMatches(out, scene, detections[0]);
    imshow("visualize.drawMatches", out);
}

TEST_F(visualize, drawDetection) {
    vector<Detection> detections = detector.detect(scene);
    ASSERT_GE(detections.size(), 1);
    Mat out = drawScene(scene);
    drawDetection(out, detections[0]);
    imshow("visualize.drawDetection", out);
}

TEST_F(visualize, blend) {
    Mat h = readHomography(PROJECT_BINARY_DIR + "/data/blokus/002.yml");
    Mat src = imread(PROJECT_BINARY_DIR + "/data/blokus/ref.jpg");
    Mat dst = imread(PROJECT_BINARY_DIR + "/data/blokus/002.jpg");
    Mat out = blend(src, dst, h);
    imshow("visualize.blend", out);
}

TEST_F(visualize, drawModelContours) {
    Mat out = imread(PROJECT_BINARY_DIR + "/data/taco/002.jpg");
    Mat h = readHomography(PROJECT_BINARY_DIR + "/data/taco/002.yml");
    drawModelContour(out, detector.modelbase().models[0], h, "taco");
    imshow("visualize.drawModelContours", out);
}
