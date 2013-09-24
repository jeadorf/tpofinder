/**
 * Copyright (c) 2012 Andreas Heider, Julius Adorf, Markus Grimm
 *
 * MIT License (http://www.opensource.org/licenses/mit-license.php)
 */

#ifndef PROVIDE_H
#define	PROVIDE_H

#include <atomic>
#include <condition_variable>
#include <opencv2/highgui/highgui.hpp>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace tpofinder {

    class ImageProvider {

      public:
        
        ImageProvider() = default;

        ImageProvider(const ImageProvider&) = delete;

        virtual ~ImageProvider() {}

        virtual bool next(cv::Mat &image) = 0;

    };

    class WebcamImageProvider : public ImageProvider {

        void capture_loop();

      public:

        WebcamImageProvider();

        ~WebcamImageProvider();

        bool next(cv::Mat &image);

      private:

        cv::VideoCapture capture_;

        std::mutex image_mutex_;

        cv::Mat image_;

        std::condition_variable ready_;

        std::atomic<bool> stop_;

        std::thread *worker_;

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
