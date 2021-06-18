#pragma once

#include <iostream>
#include <string>
#include <vector>

//  general includes
#include <fstream>

//  ros includes
#include <ros/ros.h>
#include "ros/time.h"
#include <trajectory_msgs/JointTrajectory.h>
// #include <control_msgs/FollowJointTrajectoryActionGoal.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <actionlib/client/simple_action_client.h>

class RobotGripper{

    private:
        // Action client for the joint trajectory action 
        // used to trigger the arm movement action
        actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> *traj_client_;
        std::string _actionTopic;
        std::vector<std::string> _v_joints;

    public:
        RobotGripper(const std::string &,const std::vector<std::string> );
        ~RobotGripper();

        void startTrajectory(control_msgs::FollowJointTrajectoryActionGoal ActionGoal);
        control_msgs::FollowJointTrajectoryActionGoal gripperExtensionTrajectoryPos(const std::vector<double> &pos);
        actionlib::SimpleClientGoalState getState();

};