cmake_minimum_required(VERSION 2.8.3)
project(kinova_grasp)

set (CMAKE_CXX_STANDARD 11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
             COMPONENTS
             moveit_core
             moveit_ros_planning
             moveit_ros_planning_interface
             pluginlib
             cmake_modules
             geometric_shapes
	     cv_bridge

	     message_generation
	     geometry_msgs
	     sensor_msgs
)

add_service_files(
  FILES
  Grasp.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS
    moveit_core
    moveit_ros_planning_interface
    interactive_markers
    message_runtime
)

find_package(Eigen3 REQUIRED)
find_package(kinova_driver REQUIRED)
find_package(Boost REQUIRED system filesystem date_time thread)

###########
## Build ##
###########
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

include_directories(SYSTEM ${Boost_INCLUDE_DIR} ${EIGEN_INCLUDE_DIRS})
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_LIBRARY_DIRS})

include_directories(${kinova_driver_INCLUDE_DIRS})

SET(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(robust_grasp src/robust_grasp.cpp include/robust_grasp.h)

target_link_libraries( robust_grasp ${OpenCV_LIBS} )

target_link_libraries(robust_grasp ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${kinova_driver_LIBRARIES})
install(TARGETS robust_grasp DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
