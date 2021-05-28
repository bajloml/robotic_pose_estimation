//  general includes
#include <iostream>
#include <fstream>
#include <string>

//  ros includes
#include <tf/transform_listener.h>
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

    tf::TransformListener listener_;
    tf::StampedTransform transform_;

    Eigen::Transform<float, 3, Eigen::Affine> Transformation;

    public:
        PoseSaver(std::string fPe_yamlPoseFileSavePath);
        ~PoseSaver();

        Eigen::Transform<float, 3, Eigen::Affine> saveYamlPoseFile(std::string, std::string, unsigned);
        Eigen::Transform<float, 3, Eigen::Affine> tfToEigenTransform(tf::StampedTransform);
      
};