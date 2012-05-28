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

#include "tpofinder/detect.h"
#include "tpofinder/visualize.h"

using namespace cv;
using namespace tpofinder;
using namespace std;
namespace po = boost::program_options;

const string NAME = "tpofind";

bool verbose = false;
bool webcam = false;
vector<string> files;

VideoCapture *capture = NULL;

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

void openCamera() {
    capture = new VideoCapture(0); // open the default camera
    if (!capture->isOpened()) { // check if we succeeded
        cerr << "Could not open default camera." << endl;
        exit(-1);
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

int main(int argc, char* argv[]) {
    processCommandLine(argc, argv);

    namedWindow(NAME, 1);

    // TODO: adapt to OpenCV 2.4.
    // TODO: remove duplication
    Ptr<FeatureDetector> fd = new OrbFeatureDetector(1000, 1.2, 8);
    Ptr<FeatureDetector> trainFd = new OrbFeatureDetector(250, 1.2, 8);
    Ptr<DescriptorExtractor> de = new OrbDescriptorExtractor(1000, 1.2, 8);
    Ptr<flann::IndexParams> indexParams = new flann::LshIndexParams(15, 12, 2);
    Ptr<DescriptorMatcher> dm = new FlannBasedMatcher(indexParams);

    Feature trainFeature(trainFd, de, dm);

    Modelbase modelbase(trainFeature);

    loadModel(modelbase, "data/adapter");
    loadModel(modelbase, "data/blokus");
    loadModel(modelbase, "data/stockholm");
    loadModel(modelbase, "data/taco");
    loadModel(modelbase, "data/tea");

    Feature feature(fd, de, dm);

    Ptr<DetectionFilter> filter = new AndFilter(
            Ptr<DetectionFilter > (new EigenvalueFilter(-1, 4.0)),
            Ptr<DetectionFilter > (new InliersRatioFilter(0.30)));

    Detector detector(modelbase, feature, filter);

    if (webcam) {
        openCamera();
    }

    int i = 0;
    while (webcam || i < files.size()) {
        Mat image;

        if (webcam) {
            *capture >> image;
        } else if (files.size() > 0) {
            image = imread(files[i]);
            i++;
        } else {
            string s;
            getline(cin, s);
            image = imread(s);
        }

        if (!image.empty()) {
            Scene scene = detector.describe(image);

            vector<Detection> detections = detector.detect(scene);

            BOOST_FOREACH(Detection d, detections) {
                drawDetection(image, d);
            }
        }

        imshow(NAME, image);

        if (waitKey(1) >= 0) break;
    }

    if (files.size() > 0) {
        waitKey(-1);
    }

    delete capture;

    return 0;
}
