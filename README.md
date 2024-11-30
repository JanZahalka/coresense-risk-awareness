# coresense-risk-awareness
CoreSense Risk awareness  module


# Python installs
Set up the virtual environment in the root dir (here called `env_cram`) and activate it
    virtualenv env_cram
    source env_cram/bin/activate


# System installs
Rosbag 2
    sudo apt install ros-<ros_distro>-rosbag2 ros-<ros_distro>-rosbag2-storage-default-plugins

ROS 2 CV Bridge
    sudo apt install ros-<ros_distro>-cv-bridge

Rosbag 2 - Py (in cas it didn't get installed with Rosbag 2)
    sudo apt install ros-<ros_distro>-rosbag2-py