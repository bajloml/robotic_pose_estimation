#include "wrapper_libraries/imageSaver.h"


ImageSaver::ImageSaver(std::string cameraTopicSubscribeTo, std::string imageSavePath): it_(nh_){
    cameraTopicSubscribeTo_ =   cameraTopicSubscribeTo;
    imageSavePath_          =   imageSavePath;
    imageName_              =   "image-";

    // Subscribe to input video feed
    image_sub_ = it_.subscribe(cameraTopicSubscribeTo_.c_str(), 1, &ImageSaver::imageCb, this);
    ROS_INFO("ImageSaver subscribed to topic: %s",cameraTopicSubscribeTo_.c_str());
}

ImageSaver::~ImageSaver(){
}

void ImageSaver::saveImage(unsigned imageNumber){

    imageNumber_ = imageNumber;
    imageStringStream << imageSavePath_ + imageName_ + std::to_string(imageNumber_) + ".png";
    ROS_INFO("image save path: %s", imageStringStream.str().c_str());
    
    // save the image
    saved_ = (cv::imwrite(imageStringStream.str(), cv_ptr->image));    //ROS_ASSERT();
    if(saved_){
        ROS_INFO("image file saved: %s", imageStringStream.str().c_str());
    }
    
    // clear and empty the stream
    imageStringStream.str("");
    imageStringStream.clear();
    return;
}

//  callback --> get the image from the topic
void ImageSaver::imageCb(const sensor_msgs::ImageConstPtr& msg){
    try{
        ROS_INFO("reading the image from: %s", cameraTopicSubscribeTo_.c_str());
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        return;
    }
    catch (cv_bridge::Exception& e){
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

}
