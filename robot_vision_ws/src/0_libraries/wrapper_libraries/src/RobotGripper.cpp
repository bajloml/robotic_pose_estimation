#include "wrapper_libraries/RobotGripper.h"

  //public:
    //! Initialize the action client and wait for action server to come up
    RobotGripper::RobotGripper(const std::string &actionTopic,const std::vector<std::string> v_joints):
                                                                _actionTopic(actionTopic),
                                                                _v_joints(v_joints) {

        // tell the action client that we want to spin a thread by default
        traj_client_ = new actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction>(_actionTopic, true);

        // wait for action server to come up
        while(!traj_client_->waitForServer(ros::Duration(5.0))){
            ROS_INFO("Waiting for the joint_trajectory_action server");
        }
    }

    //! Clean up the action client
    RobotGripper::~RobotGripper(){
        delete traj_client_;
    }

    //! Sends the command to start a given trajectory
    void RobotGripper::startTrajectory(control_msgs::FollowJointTrajectoryActionGoal ActionGoal){
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
    control_msgs::FollowJointTrajectoryActionGoal RobotGripper::gripperExtensionTrajectoryPos(const std::vector<double> &pos){

        //our goal variable
        control_msgs::FollowJointTrajectoryActionGoal ActionGoal;

        // number of points
        int numPoints = 1;

        // We will have two waypoints in this goal trajectory
        ActionGoal.goal.trajectory.points.resize(numPoints);
        ActionGoal.goal.trajectory.header.frame_id = "";

        for(int i=0; i<numPoints; i++){
            ActionGoal.goal.trajectory.points[i].positions.resize(_v_joints.size());
            ActionGoal.goal.trajectory.points[i].velocities.resize(_v_joints.size());
            ActionGoal.goal.trajectory.points[i].accelerations.resize(_v_joints.size());
            ActionGoal.goal.trajectory.points[i].effort.resize(_v_joints.size());
        }

        // First, the joint names, which apply to all waypoints
        ActionGoal.goal.trajectory.joint_names = _v_joints;

        // First trajectory point
        // First point velocities and positions
        for (int i = 0; i<_v_joints.size(); i++){
            ActionGoal.goal.trajectory.points[0].velocities[i] = 0.0;
        }

        ActionGoal.goal.trajectory.points[0].positions[0] = pos.at(0);
        ActionGoal.goal.trajectory.points[0].positions[1] = pos.at(1);
        ActionGoal.goal.trajectory.points[0].positions[2] = pos.at(2);

        // To be reached 1 second after starting along the trajectory
        ActionGoal.goal.trajectory.points[0].time_from_start = ros::Duration(0.5);

        //we are done; return the goal
        return ActionGoal;
    }

    //! Returns the current state of the action
    actionlib::SimpleClientGoalState RobotGripper::getState(){
        return traj_client_->getState();
    }
//};
