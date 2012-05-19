/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#include <cstdio>
#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>

#include <tpofinder/detect.h>
#include <tpofinder/visualize.h>

using namespace cv;
using namespace tpofinder;
using namespace std;

void captureImage(VideoCapture& capture, Mat& image) {
    capture >> image;
}

int main(int argc, char* argv[]) {
    VideoCapture capture(0); // open the default camera
    if (!capture.isOpened()) { // check if we succeeded
        return -1;
    }

    namedWindow("webcam", 1);

    // TODO: adapt to OpenCV 2.4.
    // TODO: remove duplication
    Ptr<FeatureDetector> fd = new OrbFeatureDetector(1000, 1.2, 8);
    Ptr<FeatureDetector> trainFd = new OrbFeatureDetector(250, 1.2, 8);
    Ptr<DescriptorExtractor> de = new OrbDescriptorExtractor(1000, 1.2, 8);
    Ptr<flann::IndexParams> indexParams = new flann::LshIndexParams(15, 12, 2);
    Ptr<DescriptorMatcher> dm = new FlannBasedMatcher(indexParams);

    Feature trainFeature(trainFd, de, dm);

    Modelbase modelbase(trainFeature);
    
    modelbase.add("data/adapter");
    modelbase.add("data/blokus");
    modelbase.add("data/stockholm");
    modelbase.add("data/taco");
    modelbase.add("data/tea");

    Feature feature(fd, de, dm);

    Ptr<DetectionFilter> filter = new AndFilter(
            Ptr<DetectionFilter > (new EigenvalueFilter(-1, 4.0)),
            Ptr<DetectionFilter > (new InliersRatioFilter(0.30)));

    Detector detector(modelbase, feature, filter);

    while (true) {
        Mat image;

        captureImage(capture, image);

        if (!image.empty()) {
            Scene scene = detector.describe(image);

            vector<Detection> detections = detector.detect(scene);

            BOOST_FOREACH(Detection d, detections) {
                drawDetection(image, d);
            }
        }
        imshow("webcam", image);

        if (waitKey(1) >= 0) break;
    }

    return 0;
}
