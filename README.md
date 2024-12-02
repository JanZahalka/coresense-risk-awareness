# coresense-risk-awareness
CoreSense Risk awareness  module


# System installs
Rosbag 2
    sudo apt install ros-<ros_distro>-rosbag2 ros-<ros_distro>-rosbag2-storage-default-plugins

ROS 2 CV Bridge
    sudo apt install ros-<ros_distro>-cv-bridge

Rosbag 2 - Py (in cas it didn't get installed with Rosbag 2)
    sudo apt install ros-<ros_distro>-rosbag2-py


# Python installs
Set up the virtual environment in the root dir (here called `env_cram`) and activate it
    virtualenv env_cram
    source env_cram/bin/activate

Install the required Python packages with `pip` 
    pip install -r requirements.txt

Link the system's `rosbag2_py` to the virtualenv
    echo "/opt/ros/rolling/lib/python3.x/site-packages" > ~/env_cram/lib/python3.x/site-packages/ros2.pth