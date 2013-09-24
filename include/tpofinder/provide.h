/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#ifndef PROVIDE_H
#define	PROVIDE_H

#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <vector>

namespace tpofinder {

    class ImageProvider {

      public:

        // TODO: prevent copying instances of this class

        virtual ~ImageProvider() {}

        virtual bool next(cv::Mat &image) = 0;

    };

    class WebcamImageProvider : public ImageProvider {

      public:

        WebcamImageProvider();

        ~WebcamImageProvider();

        bool next(cv::Mat &image);

      private:

        cv::VideoCapture *capture_;

    };

    class StdinFilenameImageProvider : public ImageProvider {

      public:

        bool next(cv::Mat &image);

    };

    class ListFilenameImageProvider : public ImageProvider {

      public:

        ListFilenameImageProvider(const std::vector<std::string> &files);

        bool next(cv::Mat &image);

      private:

        int num_;

        std::vector<std::string> files_;

    };

}

#endif
