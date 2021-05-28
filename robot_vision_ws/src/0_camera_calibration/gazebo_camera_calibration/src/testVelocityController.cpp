//  general includes
#include <iostream>
#include <fstream>
#include <string>

//  ros includes
#include <ros/ros.h>
#include "ros/time.h"
#include <trajectory_msgs/JointTrajectory.h>
// #include <control_msgs/FollowJointTrajectoryActionGoal.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <actionlib/client/simple_action_client.h>

class RobotArm{
  private:
    // Action client for the joint trajectory action 
    // used to trigger the arm movement action
    actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> *traj_client_;
    std::string _actionTopic;

  public:
    //! Initialize the action client and wait for action server to come up
    RobotArm(const std::string &actionTopic): _actionTopic(actionTopic){
    // tell the action client that we want to spin a thread by default
    traj_client_ = new actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction>(_actionTopic, true);

    // wait for action server to come up
    while(!traj_client_->waitForServer(ros::Duration(5.0))){
      ROS_INFO("Waiting for the joint_trajectory_action server");
    }
  }

  //! Clean up the action client
  ~RobotArm(){
    delete traj_client_;
  }

  //! Sends the command to start a given trajectory
  void startTrajectory(control_msgs::FollowJointTrajectoryActionGoal ActionGoal){
    // When to start the trajectory: 1s from now
    ActionGoal.goal.trajectory.header.stamp = ros::Time::now() + ros::Duration(1.0);
    if (traj_client_->isServerConnected()) {
        traj_client_->sendGoal(ActionGoal.goal);
        ROS_INFO("client connected on: %s", _actionTopic.c_str());
    }
  }

  //! Generates a simple trajectory with two waypoints, used as an example
  /*! Note that this trajectory contains two waypoints, joined together
      as a single trajectory. Alternatively, each of these waypoints could
      be in its own trajectory - a trajectory can have one or more waypoints
      depending on the desired application.
  */
    control_msgs::FollowJointTrajectoryActionGoal armExtensionTrajectory(){

    //our goal variable
    control_msgs::FollowJointTrajectoryActionGoal ActionGoal;

    // Positions
    int ind = 0;

    // We will have two waypoints in this goal trajectory
    ActionGoal.goal.trajectory.points.resize(2);
    ActionGoal.goal.trajectory.header.frame_id = "world";

    for(int i=0; i<2; i++){
        ActionGoal.goal.trajectory.points[i].positions.resize(6);
        ActionGoal.goal.trajectory.points[i].velocities.resize(6);
        ActionGoal.goal.trajectory.points[i].accelerations.resize(6);
        ActionGoal.goal.trajectory.points[i].effort.resize(6);
    }

    // First, the joint names, which apply to all waypoints
    ActionGoal.goal.trajectory.joint_names.push_back("joint_1");
    ActionGoal.goal.trajectory.joint_names.push_back("joint_2");
    ActionGoal.goal.trajectory.joint_names.push_back("joint_3");
    ActionGoal.goal.trajectory.joint_names.push_back("joint_4");
    ActionGoal.goal.trajectory.joint_names.push_back("joint_5");
    ActionGoal.goal.trajectory.joint_names.push_back("joint_6");

    // First trajectory point
    // First point velocities and positions
    ActionGoal.goal.trajectory.points[ind].velocities[0] = 0.0;
    ActionGoal.goal.trajectory.points[ind].velocities[1] = 0.0;
    ActionGoal.goal.trajectory.points[ind].velocities[2] = 0.0;
    ActionGoal.goal.trajectory.points[ind].velocities[3] = 0.0;
    ActionGoal.goal.trajectory.points[ind].velocities[4] = 0.0;
    ActionGoal.goal.trajectory.points[ind].velocities[5] = 0.0;

    ActionGoal.goal.trajectory.points[ind].positions[0] = 0.0;
    ActionGoal.goal.trajectory.points[ind].positions[1] = 0.0;
    ActionGoal.goal.trajectory.points[ind].positions[2] = 0.0;
    ActionGoal.goal.trajectory.points[ind].positions[3] = 0.0;
    ActionGoal.goal.trajectory.points[ind].positions[4] = 0.0;
    ActionGoal.goal.trajectory.points[ind].positions[5] = 0.0;

    // To be reached 1 second after starting along the trajectory
    ActionGoal.goal.trajectory.points[ind].time_from_start = ros::Duration(0.5);

    // Second trajectory point
    // Second point velocities and positions
    ind += 1;
    ActionGoal.goal.trajectory.points[ind].velocities[0] = 0.0;
    ActionGoal.goal.trajectory.points[ind].velocities[1] = 0.0;
    ActionGoal.goal.trajectory.points[ind].velocities[2] = 0.0;
    ActionGoal.goal.trajectory.points[ind].velocities[3] = 0.0;
    ActionGoal.goal.trajectory.points[ind].velocities[4] = 0.0;
    ActionGoal.goal.trajectory.points[ind].velocities[5] = 0.0;

    ActionGoal.goal.trajectory.points[ind].positions[0] = -0.3;
    ActionGoal.goal.trajectory.points[ind].positions[1] = 0.2;
    ActionGoal.goal.trajectory.points[ind].positions[2] = -0.1;
    ActionGoal.goal.trajectory.points[ind].positions[3] = -1.2;
    ActionGoal.goal.trajectory.points[ind].positions[4] = 1.5;
    ActionGoal.goal.trajectory.points[ind].positions[5] = -0.3;
    // To be reached 2 seconds after starting along the trajectory
    ActionGoal.goal.trajectory.points[ind].time_from_start = ros::Duration(0.7);

    //we are done; return the goal
    return ActionGoal;
  }

  //! Returns the current state of the action
  actionlib::SimpleClientGoalState getState(){
    return traj_client_->getState();
  }
};


int main(int argc, char** argv){

  ros::init(argc, argv, "testVelocityController");
  ros::NodeHandle   _node;
  ros::Rate         _rate(10);
  
  RobotArm arm("/fanuc_1/fanuc_arm_controller/follow_joint_trajectory/");
  // Start the trajectory
  arm.startTrajectory(arm.armExtensionTrajectory());
  // Wait for trajectory completion
  while(!arm.getState().isDone() && ros::ok()){
    usleep(50000);
  }
  return 0;
}


