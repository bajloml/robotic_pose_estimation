#pragma once
// Include the string library
#include <string>
#include <sstream>
#include <chrono>
#include <cstdint>
#include <exception>

//ros
#include "std_msgs/String.h"
#include "ros/ros.h"
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

//moveit
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/CollisionObject.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Twist.h>
#include <moveit/robot_model_loader/robot_model_loader.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/foreach.hpp>
using boost::property_tree::ptree;

class MoveGroupMove{

    private:
        std::string _planningGroup;
        std::string _referenceFrame;
        std::string _endEffectorLink;
        std::string _robotName;
        std::string _plannerId;

        ros::NodeHandle _nh;
        moveit::planning_interface::MoveGroupInterface::Options _interfaceOptions;
        moveit::planning_interface::MoveGroupInterface _move_group;

    public:
        MoveGroupMove(const std::string&, const std::string&, const std::string&, const std::string&, const std::string&);
        ~MoveGroupMove();

        std::vector<double> getJointStates();
        robot_state::RobotStatePtr getCurrentRobotState();
        const ptree& empty_ptree();
        void moveUsingSetPoseTarget(const geometry_msgs::Pose& , moveit::planning_interface::MoveItErrorCode& );
        void moveUsingSetPoseTargets(const std::vector<geometry_msgs::Pose>& , moveit::planning_interface::MoveItErrorCode& );
        void toHome(std::string srdf_path);
        void moveUsingCartesianPath(const std::vector< geometry_msgs::Pose > &waypoints, 
                                    moveit_msgs::RobotTrajectory &trajectory_msg, 
                                    double eef_step=1.0, 
                                    double jump_threshold=0.0);
        void moveUsingGrasp( std::string ROBOT_NAME, std::string REFERENCE_FRAME, const geometry_msgs::Pose &wantedPose, 
                             const char* approachAxis, double approachDirection, const char* retreatAxis, double retreatDirection);
        void moveUsingJointState(geometry_msgs::Pose&);
};
