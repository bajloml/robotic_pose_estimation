#include "wrapper_libraries/PoseSaver.h"

PoseSaver::PoseSaver(std::string fPe_yamlPoseFileSavePath){
    fPe_yamlPoseFileSavePath_   =   fPe_yamlPoseFileSavePath;
}

PoseSaver::~PoseSaver(){
}

// get the Eigen transform to use it later for markers
Eigen::Transform<float, 3, Eigen::Affine> PoseSaver::tfToEigenTransform(tf::StampedTransform tf_transform){

    Eigen::Vector3f Tran;
    Eigen::Transform<float, 3, Eigen::Affine> Transformation;
    Transformation.setIdentity();

    Tran << tf_transform.getOrigin().getX(), tf_transform.getOrigin().getY(), tf_transform.getOrigin().getZ();
    Eigen::Quaternionf Rot_Quaternion(tf_transform.getRotation().getW(), tf_transform.getRotation().getX(), tf_transform.getRotation().getY(), tf_transform.getRotation().getZ());

    Transformation = Eigen::Translation3f(Tran);
    Transformation.rotate(Rot_Quaternion.normalized());

    return Transformation;
}

// Save pose to yaml file for each image
Eigen::Transform<float, 3, Eigen::Affine> PoseSaver::saveYamlPoseFile(std::string toFrame, std::string fromFrame, unsigned fPe_yaml_Nr){

    fPe_yamlFileNumber_ = fPe_yaml_Nr;
    toFrame_            = toFrame;
    fromFrame_          = fromFrame;

    // create string stream for the file path
    fPe_yaml_path_ss_ << fPe_yamlPoseFileSavePath_ + "pose_fPe_" + std::to_string(fPe_yamlFileNumber_) + ".yaml";

    // get the transform
    listener_.waitForTransform(toFrame_, fromFrame_,ros::Time(0), ros::Duration(1.0));
    listener_.lookupTransform(toFrame_, fromFrame_,ros::Time(0), transform_);
    ROS_INFO("transform done");

    //get the roll pitch yaw(Euler) from the quaternion
    double roll, pitch, yaw;
    tf::Matrix3x3(transform_.getRotation()).getRPY(roll, pitch, yaw);

    // write transform to file
    yamlFile.open (fPe_yaml_path_ss_.str().c_str());
    yamlFile << "rows: 6\n";
    yamlFile << "cols: 1\n";
    yamlFile << "data: \n";
    yamlFile << "  - [" << (transform_.getOrigin().getX()) <<"]\n";
    yamlFile << "  - [" << (transform_.getOrigin().getY()) <<"]\n";
    yamlFile << "  - [" << (transform_.getOrigin().getZ()) <<"]\n";
    yamlFile << "  - [" << (roll) <<"]\n";
    yamlFile << "  - [" << (pitch) <<"]\n";
    yamlFile << "  - [" << (yaw) <<"]\n";
    yamlFile.close();

    // get the eigen transform from the tf::Stamped to show it in markers later
    Transformation = tfToEigenTransform(transform_);

    ROS_INFO("pose file saved: %s", fPe_yaml_path_ss_.str().c_str());
    fPe_yaml_path_ss_.str("");
    fPe_yaml_path_ss_.clear();
    
    // return;
    return Transformation;
}