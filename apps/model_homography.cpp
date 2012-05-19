/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#include "tpofinder/util.h"
#include "tpofinder/visualize.h"

#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <cstdlib>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace tpofinder;
using namespace std;
namespace bpo = boost::program_options;

int main(int argc, char** argv) {
    // blokus 002
    //    Mat train = (Mat_<Point2f > (4, 1) <<
    //            Point2f(256, 156),
    //            Point2f(250, 288),
    //            Point2f(326, 288),
    //            Point2f(355, 150));
    //    Mat ref = (Mat_<Point2f > (4, 1) <<
    //            Point2f(207, 112),
    //            Point2f(226, 319),
    //            Point2f(416, 365),
    //            Point2f(468, 135));

    // blokus 003
    //    Mat train = (Mat_<Point2f > (4, 1) <<
    //            Point2f(266, 191),
    //            Point2f(268, 278),
    //            Point2f(359, 274),
    //            Point2f(360, 190));
    //    Mat ref = (Mat_<Point2f > (4, 1) <<
    //            Point2f(189, 94),
    //            Point2f(217, 375),
    //            Point2f(498, 376),
    //            Point2f(515, 88));

    //    // taco 002
    //    Mat train = (Mat_<Point2f > (4, 1) <<
    //            Point2f(260, 157),
    //            Point2f(268, 305),
    //            Point2f(436, 291),
    //            Point2f(442, 158));
    //    Mat ref = (Mat_<Point2f > (4, 1) <<
    //            Point2f(276, 198),
    //            Point2f(276, 348),
    //            Point2f(446, 348),
    //            Point2f(438, 212));

    string cfile, ofile;
    string tfile, rfile;

    bpo::options_description options;
    options.add_options()
            ("correspondences,c", bpo::value<string > (&cfile), "path to a file "
            "containing point correspondences. This file must be structured "
            "as follows:\n\n"
            "%YAML:1.0\n"
            "train: !!opencv-matrix\n"
            "   rows: 4\n"
            "   cols: 1\n"
            "   dt: \"2f\"\n"
            "   data: [ 260., 157., 268., 305., 436., 291., 442., 158. ]\n"
            "ref: !!opencv-matrix\n"
            "   rows: 4\n"
            "   cols: 1\n"
            "   dt: \"2f\"\n"
            "   data: [ 276., 198., 276., 348., 446., 348., 438., 212. ]\n\n"
            "Here the matrices in 'data' are of the form [x1 y1 x2 y2 ... x4 y4]; "
            "so put the points in the training image into one matrix and the "
            "corresponding points in the reference image into the other.")
            ("out,o", bpo::value<string > (&ofile), "path to a file where the computed "
            "homography shall be written to.")
            ("ref,r", bpo::value<string > (&rfile), "path to reference image (optional).")
            ("train,t", bpo::value<string > (&tfile), "path to train image (optional).");

    bpo::variables_map vm;
    bpo::store(bpo::parse_command_line(argc, argv, options), vm);
    notify(vm);

    Mat ref;
    Mat train;

    if (vm.count("correspondences") == 1) {
        FileStorage in(cfile, FileStorage::READ);
        in["ref"] >> ref;
        in["train"] >> train;
        in.release();
    } else {
        options.print(cerr);
        return -1;
    }

    Mat homography = findHomography(ref, train);
    cout << homography << endl;

    if (!ofile.empty()) {
        writeHomography(ofile, homography);
    }

    if (!rfile.empty() && !tfile.empty()) {
        Mat refImg = imread(rfile);
        Mat trainImg = imread(tfile);
        
        Mat refPointsImg = refImg.clone();
        Mat trainPointsImg = trainImg.clone();
        for (int i = 0; i < ref.rows; i++) {
            circle(refPointsImg, ref.at<Point2f>(i), 3, Scalar(255, 0, 0), CV_FILLED);
            circle(trainPointsImg, train.at<Point2f>(i), 3, Scalar(255, 0, 0), CV_FILLED);
        }
        imshow("keys on reference image", refPointsImg);
        imshow("keys on training image", trainPointsImg);

        Mat out = blend(refImg, trainImg, homography);
        string name = str(boost::format("homography: [%s] %s -> %s [%s]") % cfile % rfile % tfile % ofile);
        imshow(name, out);
        waitKey(0);
    }

    return 0;
}

