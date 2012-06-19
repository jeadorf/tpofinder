/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#include "tpofinder/truth.h"
#include "tpofinder/visualize.h"

using namespace cv;
using namespace tpofinder;
using namespace std;

int main(int /* argc */, char** /* argv */) {
    cvStartWindowThread();
    namedWindow("sequence_homography");

    HomographySequenceEstimator estimator;
    Mat firstImage;
    while (true) {
        string p;
        getline(cin, p);
        if (p.size() > 0) {
            Mat image = imread(p, 0);
            if (firstImage.empty()) {
                firstImage = image;
            }
            Mat homography = estimator.next(image);

            Mat out = blend(firstImage, image, homography);
            imshow("sequence_homography", out);
            waitKey(0);
        }
    }

    return 0;
}

