/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "tpofinder/configure.h"
#include "tpofinder/detect.h"
#include "tpofinder/provide.h"
#include "tpofinder/visualize.h"

using namespace cv;
using namespace tpofinder;
using namespace std;
namespace po = boost::program_options;

const string NAME = "tpofind";

bool verbose = false;
bool webcam = false;
vector<string> files;

void processCommandLine(int argc, char* argv[]) {
    po::options_description named_opts;
    named_opts.add_options()
            ("webcam,w", "Read images from webcam.")
            ("verbose,v", "Display verbose messages.")
            ("help,h", "Print help message.");

    po::options_description hidden_opts;
    hidden_opts.add_options()
        ("file,f", po::value<vector<string> >(&files));

    po::options_description all_opts;
    all_opts.add(named_opts);
    all_opts.add(hidden_opts);

    po::positional_options_description popts;
    popts.add("file", -1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
        .options(all_opts)
        .positional(popts)
        .run(), vm);
    po::notify(vm);

    webcam = vm.count("webcam") > 0;
    verbose = vm.count("verbose") > 0;

    if (vm.count("help")) {
        cout << "Usage: tpofind [OPTIONS] image ..." << endl;
        cout << named_opts << endl;
        exit(0);
    }
}

void loadModel(Modelbase& modelbase, const string& path) {
    if (verbose) {
        cout << boost::format("Loading object %-20s ... ") % path;
    }
    modelbase.add(path);
    if (verbose) {
        cout << "[DONE]" << endl;
    }
}

void processImage(Detector& detector, Mat &image) {
    if (!image.empty()) {
        cout << "Detecting objects on image          ... ";
        Scene scene = detector.describe(image);
        vector<Detection> detections = detector.detect(scene);
        cout << "[DONE]" << endl;

        BOOST_FOREACH(Detection d, detections) {
            drawDetection(image, d);
        }
    }
}

int main(int argc, char* argv[]) {
    processCommandLine(argc, argv);

    cvStartWindowThread();
    namedWindow(NAME, CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO);

    // TODO: remove duplication
    // TODO: support SIFT
    // TODO: make customizable
    Ptr<FeatureDetector> fd = new OrbFeatureDetector(1000, 1.2, 8);
    Ptr<FeatureDetector> trainFd = new OrbFeatureDetector(250, 1.2, 8);
    Ptr<DescriptorExtractor> de = new OrbDescriptorExtractor(1000, 1.2, 8);

    Ptr<flann::IndexParams> indexParams = new flann::LshIndexParams(15, 12, 2);
    Ptr<DescriptorMatcher> dm = new FlannBasedMatcher(indexParams);

    Feature trainFeature(trainFd, de, dm);

    Modelbase modelbase(trainFeature);

    loadModel(modelbase, PROJECT_BINARY_DIR + "/data/adapter");
    loadModel(modelbase, PROJECT_BINARY_DIR + "/data/blokus");
    loadModel(modelbase, PROJECT_BINARY_DIR + "/data/stockholm");
    loadModel(modelbase, PROJECT_BINARY_DIR + "/data/taco");
    loadModel(modelbase, PROJECT_BINARY_DIR + "/data/tea");

    Feature feature(fd, de, dm);

    Ptr<DetectionFilter> filter = new AndFilter(
            Ptr<DetectionFilter> (new EigenvalueFilter(-1, 4.0)),
            Ptr<DetectionFilter> (new InliersRatioFilter(0.30)));

    Detector detector(modelbase, feature, filter);

    ImageProvider *image_provider;
    if (webcam) {
        image_provider = new WebcamImageProvider();
    } else if (files.size() > 0) {
        image_provider = new ListFilenameImageProvider(files);
    } else {
        image_provider = new StdinFilenameImageProvider();
    }

    Mat image;
    while (image_provider->next(image)) {
        processImage(detector, image);
        if (!image.empty()) {
            imshow(NAME, image);
        }
    }

    delete image_provider;

    if (verbose) {
        cout << "No more images to process           ... [DONE]" << endl;
    }

    cout << "Waiting for key (win) or CTRL+C     ... [DONE]" << endl;
    while (waitKey(10) == -1) {
        imshow(NAME, image);
    }

    if (verbose) {
        cout << "Quitting                            ... [DONE]" << endl;
    }

    return 0;
}
