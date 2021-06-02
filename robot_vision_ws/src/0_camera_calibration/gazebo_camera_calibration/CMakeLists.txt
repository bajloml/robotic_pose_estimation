cmake_minimum_required(VERSION 2.8.3)
project(gazebo_camera_calibration)

# message( ${CMAKE_CURRENT_SOURCE_DIR} )

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  tf
  wrapper_libraries
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES gazebo_camera_calibration
  CATKIN_DEPENDS 
  roscpp
  rospy 
  std_msgs
  wrapper_libraries
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/gazebo_camera_calibration_node.cpp)

add_executable(getPoseImage_node src/getPoseImage_node.cpp)
target_link_libraries(getPoseImage_node ${catkin_LIBRARIES})

add_executable(intrinsic_calibration_node src/intrinsic_calibration.cpp)
target_link_libraries(intrinsic_calibration_node ${catkin_LIBRARIES})

add_executable(testVelocityController src/testVelocityController.cpp)
target_link_libraries(testVelocityController ${catkin_LIBRARIES})