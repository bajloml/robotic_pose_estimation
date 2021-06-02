//  general includes
#include <iostream>
#include <fstream>
#include <string>

//  ros includes
//#include <tf/transform_listener.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/convert.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>

//  eigen transformation
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

class PoseSaver{
    // local variable
    std::string fPe_yamlPoseFileSavePath_;
    std::string fPe_yamlFileName_;
    std::stringstream fPe_yaml_path_ss_;
    std::string toFrame_;
    std::string fromFrame_;
    unsigned fPe_yamlFileNumber_;
    bool saved_;
    std::ofstream yamlFile;

    tf2_ros::Buffer tfBuffer;
    tf2_ros::TransformListener listener_;
    geometry_msgs::TransformStamped transform_;

    public:
        PoseSaver(std::string fPe_yamlPoseFileSavePath);
        ~PoseSaver();

        geometry_msgs::Transform saveYamlPoseFile(std::string, std::string, unsigned);
        Eigen::Transform<float, 3, Eigen::Affine> tfToEigenTransform(geometry_msgs::TransformStamped);
      
};