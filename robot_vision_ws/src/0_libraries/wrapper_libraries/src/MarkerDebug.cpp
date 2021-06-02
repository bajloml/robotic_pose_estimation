#include "wrapper_libraries/MarkerDebug.h"


MarkerDebug::MarkerDebug(std::string referenceFrame):
    _visual_tools(referenceFrame){
        _referenceFrame = referenceFrame;
        _visual_tools.loadRemoteControl();
        _text_pose = Eigen::Isometry3d::Identity();
}


MarkerDebug::~MarkerDebug(){
}

// add marker at the pose
void MarkerDebug::addMarker(geometry_msgs::Pose pose, std::string markerText){
    // publish reference frame axis
    _visual_tools.publishText(_text_pose, _referenceFrame, rviz_visual_tools::WHITE, rviz_visual_tools::XLARGE);
    // publish frame and the label
    _visual_tools.publishAxisLabeled(pose, markerText, rviz_visual_tools::XSMALL);
    _visual_tools.trigger();
}

// delete all markers
void MarkerDebug::deleteMarkers(){

    _visual_tools.deleteAllMarkers();
    _visual_tools.trigger();
}

// switch from Eigen Transform to geometry msg pose
geometry_msgs::Pose MarkerDebug::EigenTransToPose(Eigen::Transform<float, 3, Eigen::Affine> transform){

    geometry_msgs::Pose pose;
    Eigen::Quaternionf quaternion;

    pose.position.x = transform.translation()[0];
    pose.position.y = transform.translation()[1];
    pose.position.z = transform.translation()[2];

    quaternion = transform.rotation();

    pose.orientation.x = quaternion.derived().coeffs()[0];
    pose.orientation.y = quaternion.derived().coeffs()[1];
    pose.orientation.z = quaternion.derived().coeffs()[2];
    pose.orientation.w = quaternion.derived().coeffs()[3];

    return pose;
}

// switch from Eigen Transform to geometry msg pose
geometry_msgs::Pose MarkerDebug::geometryTransformToPose(geometry_msgs::Transform transform){

    geometry_msgs::Pose pose;

    pose.position.x = transform.translation.x;
    pose.position.y = transform.translation.y;
    pose.position.z = transform.translation.z;

    pose.orientation.x = transform.rotation.x;
    pose.orientation.y = transform.rotation.y;
    pose.orientation.z = transform.rotation.z;
    pose.orientation.w = transform.rotation.w;

    return pose;
}
