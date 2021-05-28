//  general includes
#include <iostream>
#include <fstream>
#include <string>

//  ros includes
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class ImageSaver{

    // local variable
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    std::string cameraTopicSubscribeTo_;
    std::string imageSavePath_;
    std::string imageName_;
    std::stringstream imageStringStream;
    cv_bridge::CvImagePtr cv_ptr;
    unsigned imageNumber_;
    bool saved_;

    public:
        ImageSaver(std::string , std::string ); 
        ~ImageSaver();

        void saveImage(unsigned);
        void imageCb(const sensor_msgs::ImageConstPtr&);
        
};