//  general includes
#include <iostream>
#include <fstream>
#include <string>

//  ros includes
#include <tf/transform_listener.h>
#include <geometry_msgs/Pose.h>
#include <ros/ros.h>

//  eigen transformation
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

//moveit
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <rviz_visual_tools/rviz_visual_tools.h>

class MarkerDebug{

    std::string _referenceFrame;
    
    // moveit visual tools will publish under the topic "rviz_visual_tools" if is call in not namespace,
    // in a case of namespace, the topic will be "ns/rviz_visual_tools"
    moveit_visual_tools::MoveItVisualTools _visual_tools; 
    
    Eigen::Isometry3d _text_pose;

    public:

        MarkerDebug(std::string reference_frame);
        ~MarkerDebug();

        void addMarker(geometry_msgs::Pose pose, std::string markerText);
        void deleteMarkers();

        geometry_msgs::Pose EigenTransToPose(Eigen::Transform<float, 3, Eigen::Affine> transform);
        geometry_msgs::Pose geometryTransformToPose(geometry_msgs::Transform transform);

};