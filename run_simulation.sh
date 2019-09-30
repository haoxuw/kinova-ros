PROCSTRING='gzserver|roslaunch|controller_manager|roslaunch|rosmaster'
PROCS=`ps -ef | egrep $PROCSTRING | grep -v grep`
if [[ ! -z ${PROCS} ]]; then
    echo "Found the following processes related to $PROCSTRING"
    echo 
    echo "Termintating.."
    set -x
    echo $PROCS | awk '{print $2}' | xargs kill -s SIGINT
    echo $PROCS | awk '{print $2}' | xargs kill -9
fi
rosclean purge -y

sleep 5

roslaunch kinova_gazebo jacam_launch.launch kinova_robotType:=j2s7s300 -v --screen &





#catkin_make && source devel/setup.bash && rosrun kinova_grasp robust_grasp

#roslaunch kinova_gazebo jacam_launch.launch kinova_robotType:=j2s7s300 -v --screen

#rosrun kinova_grasp robust_grasp