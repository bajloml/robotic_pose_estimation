#include "wrapper_libraries/PoseSaver.h"

PoseSaver::PoseSaver(std::string fPe_yamlPoseFileSavePath):listener_(tfBuffer){
    fPe_yamlPoseFileSavePath_ = fPe_yamlPoseFileSavePath;
    tfBuffer.setUsingDedicatedThread(true);
}

PoseSaver::~PoseSaver(){
}

// get the Eigen transform to use it later for markers
// Eigen::Transform<float, 3, Eigen::Affine> PoseSaver::tfToEigenTransform(tf::StampedTransform tf_transform){
Eigen::Transform<float, 3, Eigen::Affine> PoseSaver::tfToEigenTransform(geometry_msgs::TransformStamped tf_transform){

    Eigen::Vector3f Tran;
    Eigen::Transform<float, 3, Eigen::Affine> Transformation;
    Transformation.setIdentity();

    // Tran << tf_transform.getOrigin().getX(), tf_transform.getOrigin().getY(), tf_transform.getOrigin().getZ();
    // Eigen::Quaternionf Rot_Quaternion(tf_transform.getRotation().getW(), tf_transform.getRotation().getX(), tf_transform.getRotation().getY(), tf_transform.getRotation().getZ());

    
    Tran << tf_transform.transform.translation.x, tf_transform.transform.translation.y, tf_transform.transform.translation.z;
    Eigen::Quaternionf Rot_Quaternion(tf_transform.transform.rotation.w,
                                      tf_transform.transform.rotation.x, 
                                      tf_transform.transform.rotation.y, 
                                      tf_transform.transform.rotation.z);



    Transformation = Eigen::Translation3f(Tran);
    Transformation.rotate(Rot_Quaternion.normalized());

    return Transformation;
}

// Save pose to yaml file for each image
geometry_msgs::Transform PoseSaver::saveYamlPoseFile(std::string toFrame, std::string fromFrame, unsigned fPe_yaml_Nr){

    geometry_msgs::Transform Transformation;

    fPe_yamlFileNumber_ = fPe_yaml_Nr;
    toFrame_            = toFrame;
    fromFrame_          = fromFrame;

    // create string stream for the file path
    fPe_yaml_path_ss_ << fPe_yamlPoseFileSavePath_ + "pose_fPe_" + std::to_string(fPe_yamlFileNumber_) + ".yaml";

    // get the transform
    // listener_.waitForTransform(toFrame_, fromFrame_,ros::Time(0), ros::Duration(1.0));
    // listener_.lookupTransform(toFrame_, fromFrame_,ros::Time(0), transform_);
    std::string * transform_error;

    if (tfBuffer.canTransform(toFrame_, fromFrame_, ros::Time(0), ros::Duration(3)), transform_error){
        transform_ = tfBuffer.lookupTransform(toFrame_, fromFrame_, ros::Time(0));
        Transformation = transform_.transform;
        ROS_INFO("transform done");

        // convert msg quaterion to tf2 quaternion
        tf2::Quaternion tf2_quaternion(transform_.transform.rotation.x,
                                       transform_.transform.rotation.y,
                                       transform_.transform.rotation.z,
                                       transform_.transform.rotation.w);

        double roll, pitch, yaw;
        tf2::Matrix3x3 m(tf2_quaternion);
        m.getRPY(roll, pitch, yaw);

        // write transform to file
        yamlFile.open (fPe_yaml_path_ss_.str().c_str());
        yamlFile << "rows: 6\n";
        yamlFile << "cols: 1\n";
        yamlFile << "data: \n";
        yamlFile << "  - [" << (transform_.transform.translation.x)<<"]\n";
        yamlFile << "  - [" << (transform_.transform.translation.y)<<"]\n";
        yamlFile << "  - [" << (transform_.transform.translation.z) <<"]\n";
        yamlFile << "  - [" << (roll) <<"]\n";
        yamlFile << "  - [" << (pitch) <<"]\n";
        yamlFile << "  - [" << (yaw) <<"]\n";
        yamlFile.close();

    }

    ROS_INFO("pose file saved: %s", fPe_yaml_path_ss_.str().c_str());
    fPe_yaml_path_ss_.str("");
    fPe_yaml_path_ss_.clear();
    
    // return;
    return Transformation;
}