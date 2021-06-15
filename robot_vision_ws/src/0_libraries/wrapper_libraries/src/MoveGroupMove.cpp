
#include "wrapper_libraries/MoveGroupMove.h"
//pugixml parser
#include <pugixml.hpp>
//moveit shapes
#include <geometric_shapes/shape_operations.h>

MoveGroupMove::MoveGroupMove(const std::string &planningGroup,
                             const std::string &referenceFrame,
                             const std::string &endEffectorLink,
                             const std::string &robotName,
                             const std::string &plannerId):  
    _move_group(_interfaceOptions),
    _interfaceOptions( _planningGroup, _robotName + "/robot_description", _nh), // robot desription under the namespace, under which the actions servers are running
    _nh(_robotName),            
    _planningGroup{planningGroup},
    _referenceFrame{referenceFrame},
    _endEffectorLink{endEffectorLink}, 
    _robotName{robotName},
    _plannerId{plannerId}{

        _move_group.setPlanningTime(10);
        _move_group.setNumPlanningAttempts(2);
        _move_group.allowReplanning(true);
        
        //set EEF link
        /*_move_group.setEndEffectorLink(_endEffectorLink); */
        
        _move_group.setGoalPositionTolerance(0.001);
        _move_group.setGoalOrientationTolerance(0.001);
        
        _move_group.setPlannerId(_plannerId);
    }

MoveGroupMove::~MoveGroupMove(){}

std::vector<double> MoveGroupMove::getJointStates(){

    return _move_group.getCurrentJointValues();
}

robot_state::RobotStatePtr MoveGroupMove::getCurrentRobotState(){

    return _move_group.getCurrentState();
}

void MoveGroupMove::moveUsingSetPoseTarget(const geometry_msgs::Pose &pose, moveit::planning_interface::MoveItErrorCode &error_code){

    moveit::core::RobotStatePtr robotStatePtr = _move_group.getCurrentState();
    const robot_state::JointModelGroup *joint_model_group = robotStatePtr->getJointModelGroup(_planningGroup);

    _move_group.setPoseTarget(pose);
    //_move_group.setJointValueTarget(pose, _endEffectorLink);

    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    bool successPlan = (_move_group.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if(successPlan){
        error_code = _move_group.move();
    }
    else{
        ROS_WARN("move group, unsuccessful planing");
    }
}

void MoveGroupMove::moveUsingSetPoseTargets(const std::vector<geometry_msgs::Pose> &poseVector, moveit::planning_interface::MoveItErrorCode &error_code){

    moveit::core::RobotStatePtr robotStatePtr = _move_group.getCurrentState();
    const robot_state::JointModelGroup *joint_model_group = robotStatePtr->getJointModelGroup(_planningGroup);

    _move_group.setPoseTargets(poseVector);
    //_move_group.setJointValueTarget(pose, _endEffectorLink);

    moveit::planning_interface::MoveGroupInterface::Plan my_plan;
    bool successPlan = (_move_group.plan(my_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);

    if(successPlan){
        error_code = _move_group.move();
    }
    else{
        ROS_WARN("move group, unsuccessful planing");
    }
}

void MoveGroupMove::moveUsingCartesianPath(const std::vector< geometry_msgs::Pose > &waypoints,
                                           moveit_msgs::RobotTrajectory &trajectory_msg, 
                                           double eef_step=0.0, 
                                           double jump_threshold=1.0){

    moveit::core::RobotStatePtr robotStatePtr = _move_group.getCurrentState();
    const robot_state::JointModelGroup *joint_model_group = robotStatePtr->getJointModelGroup(_planningGroup);
    //_move_group.setPoseTarget(pose);
    //const std::vector< geometry_msgs::Pose > &waypoints, double eef_step, double jump_threshold, moveit_msgs::RobotTrajectory &trajectory, bool avoid_collisions=true, moveit_msgs::MoveItErrorCodes *error_code=NULL
    double fraction = _move_group.computeCartesianPath( waypoints,
                                                        eef_step,
                                                        jump_threshold,
                                                        trajectory_msg, 
                                                        false);
    ROS_WARN("Cartesian trajectory was succesfully planned %.2f%%", fraction * 100.0);                                                    
    if(fraction==1.0){
        ROS_WARN("Cartesian planning was succesfull. Moving...");
        _move_group.execute(trajectory_msg);
    }
    else{
        ROS_WARN("Cartesian planning was unsuccesful!");
    }
}

//approach to target using moveit grasp
void MoveGroupMove::moveUsingGrasp( std::string ROBOT_NAME, 
                                    std::string REFERENCE_FRAME, 
                                    const geometry_msgs::Pose &wantedPose, 
                                    const char* approachAxis,
                                    double approachDirection,
                                    const char* retreatAxis,
                                    double retreatDirection)
{
    std::vector<moveit_msgs::Grasp> grasps;
    grasps.resize(1);

    grasps[0].grasp_pose.header.frame_id = ROBOT_NAME + "/" + REFERENCE_FRAME;
    grasps[0].grasp_pose.pose.orientation = wantedPose.orientation;
    grasps[0].grasp_pose.pose.position = wantedPose.position;

    /* Set approach parameters */
    /* Defined with respect to frame_id */
    grasps[0].pre_grasp_approach.direction.header.frame_id = ROBOT_NAME + "/" + REFERENCE_FRAME;
    
    if (approachDirection!=1.0 && approachDirection !=-1.0)
        return;

    switch (*approachAxis)
    {
        case 'x':
        case 'X':
            grasps[0].pre_grasp_approach.direction.vector.x = approachDirection;
            break;
        case 'y':
        case 'Y':
            grasps[0].pre_grasp_approach.direction.vector.y = approachDirection;
            break;
        case 'z':
        case 'Z':
            grasps[0].pre_grasp_approach.direction.vector.z = approachDirection;
            break;
        default:
            std::cout << "approachAxis must be set to x,y or z " << std::endl;
            return;
    }

    grasps[0].pre_grasp_approach.min_distance = 0.1;      //0.095
    grasps[0].pre_grasp_approach.desired_distance = 0.15;   //0.215

    /* Set reterat parameters */
    /* Defined with respect to frame_id */
    grasps[0].post_grasp_retreat.direction.header.frame_id =  ROBOT_NAME + "/" + REFERENCE_FRAME;

    if (retreatDirection!=1.0 && retreatDirection !=-1.0)
        return;

    switch (*retreatAxis)
    {
        case 'x':
        case 'X':
            grasps[0].post_grasp_retreat.direction.vector.x = retreatDirection;
            break;
        case 'y':
        case 'Y':
            grasps[0].post_grasp_retreat.direction.vector.y = retreatDirection;
            break;
        case 'z':
        case 'Z':
            grasps[0].post_grasp_retreat.direction.vector.z = retreatDirection;
            break;
        default:
            std::cout << "retreatAxis must be set to x,y or z " << std::endl;
            return;
    }
    // Setting post-grasp retreat
    // ++++++++++++++++++++++++++

    grasps[0].post_grasp_retreat.min_distance = 0.1;
    grasps[0].post_grasp_retreat.desired_distance = 0.15;

    // Set support surface as table1.
    _move_group.setSupportSurfaceName("table1");
    // Call pick to pick up the object using the grasps given
    _move_group.pick("object", grasps);
}

//read "home" position from srdf file and send robot there
void MoveGroupMove::toHome(std::string srdf_path){   
    const std::string val = "value";
    const char* val_c = val.c_str();

    std::vector<double> joints;
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(srdf_path.c_str());
    if (!result){
        std::cout << "Given SRDF file doesn't exist!!!" << std::endl;
        return;
    }
    else{
        std::cout << "Given SRDF file exist" << std::endl;
    }
    pugi::xml_node robot = doc.child("robot");
    pugi::xml_node group_state = robot.child("group_state");
    pugi::xml_node home_pose_node;

    //first find child group state with name "home"
    for (pugi::xml_node node = robot.first_child(); node; node = node.next_sibling()){
        //if node name is "home" and node attribute name is "group_state" then we have correct node
        if(std::string(node.name())=="group_state" && std::string(node.first_attribute().value()) == "home"){
            home_pose_node=node;
            break;
        }
    }
    //fill vector with joint values from srdf file
    for (pugi::xml_node joint = home_pose_node.first_child(); joint; joint = joint.next_sibling()){
        std::cout << joint.name() << ": " << joint.value() << std::endl;
        if(std::string(joint.name())=="joint"){
                const pugi::char_t *val = joint.attribute(val_c).value();
                joints.push_back(atof(val));
        }
    }
    std::cout << std::endl;

    //send move group to home position
    _move_group.setJointValueTarget(joints);
    _move_group.move();
}

void MoveGroupMove::moveUsingJointState(geometry_msgs::Pose &pose){

    robot_model_loader::RobotModelLoader robot_model_loader(_robotName + "/robot_description");
    robot_model::RobotModelPtr robot_model = robot_model_loader.getModel();

    //Create a RobotState and JointModelGroup to keep track of the current robot pose and planning group
    robot_state::RobotStatePtr robot_state(new robot_state::RobotState(robot_model));
    const robot_state::JointModelGroup* joint_model_group = robot_state->getJointModelGroup(_planningGroup);

    const std::vector<std::string>& joint_names = joint_model_group->getVariableNames();
    std::vector<double> joint_values;

    ROS_INFO("checking inverse kinematics");
    if(robot_state->setFromIK(joint_model_group, pose)){
        ROS_INFO("IK success!");
        robot_state->copyJointGroupPositions(joint_model_group, joint_values);
        _move_group.setJointValueTarget(joint_values);
    }
    else{
        ROS_ERROR("IK FAILED");
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = (_move_group.plan(plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS);
    _move_group.execute(plan);

    return;

}


//Planning scene handler class
PlanningSceneHandler::PlanningSceneHandler( std::string referenceFrame,std::string _planningTopic):
                                            _refFrame{referenceFrame},
                                            _planningTopic{_planningTopic}{   
    ROS_INFO("CONSTRUCTING PLANNING HANDLER---------------------------------------------------------");
    _planningSceneDiffPublisher = _nodeHandle.advertise<moveit_msgs::PlanningScene>(_planningTopic, 1);
    ROS_INFO("PLANNING HANDLER ADVERTISED ON TOPIC");
}

PlanningSceneHandler::~PlanningSceneHandler(){}

//void PlanningSceneHandler::addObject( std::string _objectName, std::string _meshModelPath, geometry_msgs::Pose _collisionObjectPose)
void PlanningSceneHandler::addObject( std::string _objectName, geometry_msgs::Pose _collisionObjectPose){

    moveit_msgs::CollisionObject co;

    //shapes::Mesh* m = shapes::createMeshFromResource(_meshModelPath); 
    //ROS_INFO("mesh loaded");
    shape_msgs::Mesh co_mesh;
    shapes::ShapeMsg co_mesh_msg;  
    //shapes::constructMsgFromShape(m, co_mesh_msg);

    /*co_mesh = boost::get<shape_msgs::Mesh>(co_mesh_msg);  
    co.meshes.resize(1);
    co.mesh_poses.resize(1);
    co.meshes[0] = co_mesh;
    co.header.frame_id = _refFrame;
    co.id = _objectName;

    // add position and orientation to mesh 
    co.mesh_poses[0].position.x = _collisionObjectPose.position.x;
    co.mesh_poses[0].position.y = _collisionObjectPose.position.y;
    co.mesh_poses[0].position.z = _collisionObjectPose.position.z;
    co.mesh_poses[0].orientation.w = _collisionObjectPose.orientation.w;
    co.mesh_poses[0].orientation.x = _collisionObjectPose.orientation.x;
    co.mesh_poses[0].orientation.y = _collisionObjectPose.orientation.y;
    co.mesh_poses[0].orientation.z = _collisionObjectPose.orientation.z;  */

    //co.meshes.push_back(co_mesh);
    //co.mesh_poses.push_back(co.mesh_poses[0]);

    /* Define a box to be attached */
    shape_msgs::SolidPrimitive primitive;
    primitive.type = primitive.BOX;
    primitive.dimensions.resize(3);
    primitive.dimensions[0] = 0.1;
    primitive.dimensions[1] = 0.1;
    primitive.dimensions[2] = 0.1;

    co.header.frame_id = _refFrame;
    co.id = _objectName;

    co.primitives.push_back(primitive);
    co.primitive_poses.push_back(_collisionObjectPose);
    co.operation = co.ADD;

    //delete everything from planning scene and add collision object on the planning scene
    _planningScene.world.collision_objects.clear();
    _planningScene.world.collision_objects.push_back(co);
    _planningScene.is_diff=true;
    //using planning scene difference
    _planningSceneDiffPublisher.publish(_planningScene);
    
}



