from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='arm_tools',
            executable='topic_analyzer_node',
            name='topic_analyzer_node',
        ),
    ])