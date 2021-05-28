#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//  general includes
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

//  ros includes
#include <ros/ros.h>

class Intrinsic_Cal{

    // local variable
    std::string pathToImages_;       //  path to the dir where the images are
    std::string savePath_;           //  path to the dir where the camera_settings.json should be saved
    std::string node_name_;          //  name of the node
    bool showImages_;                //  should the image be shown during the calibration

    // dimensions of the chessboard
    int chessboard_dim_[2] = {6, 9};

    bool succ_;

    cv::Mat img_, img_gray_;                             // define image and gray image

    std::vector<cv::String> imagesPath_;                 // vector extract each image path from pathToImages

    std::vector<std::vector<cv::Point2f> > points_2D_;   // 2D points vector of vectors of type cv::Point2f to be filled with the image points
    std::vector<std::vector<cv::Point3f> > points_3D_;   // 3D points vector of vectors of type cv::Point3f to be  filled with the object points

    std::vector<cv::Point2f> coords_2D_;                 // vector for 2D (pixel) coordinates of detected chessboard corners 
    std::vector<cv::Point3f> coords_3D_;                 // vector for 3D (chessboard) coords of 3D points

    cv::Mat camMatrix_, dist_, R_, T_;                   // Matrices to be filled by the calibration function
    double RMS_;                                         // RMS error after calibration


    public:
        Intrinsic_Cal(bool showImages);
        ~Intrinsic_Cal();

        void calibrate(std::string pathToImages);
        void saveSettings(std::string savePath);
      
};