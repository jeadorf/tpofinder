#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv){
    testing::InitGoogleTest(&argc, argv);
    int s = RUN_ALL_TESTS();
    cv::waitKey(0);
    return s;
}