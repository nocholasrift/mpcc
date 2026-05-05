import os
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, EnvironmentVariable, PathJoinSubstitution, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

from launch_ros.parameter_descriptions import ParameterFile


def generate_launch_description():

    # --- Arguments ---
    raw_odom_topic = LaunchConfiguration('rawOdomTopic')
    mpc_odom_topic = LaunchConfiguration('mpcOdomTopic')
    cmd_topic = LaunchConfiguration('cmdTopic')
    param_file = LaunchConfiguration('param_file')
    db_filename = LaunchConfiguration('db_filename')
    use_sim_time = LaunchConfiguration('use_sim_time')
    slam_params_file = LaunchConfiguration('slam_params_file')

    pkg_share = get_package_share_directory('mpcc')
    juggler_layout_path = os.path.join(pkg_share, 'params', 'plotjuggler_layout.xml')
    # mpcc_param_file_path = PathJoinSubstitution([pkg_share, 'params', param_file])
    node_env = os.environ.copy()
    node_env['PYTHONUNBUFFERED'] = '1'

    return LaunchDescription([

        DeclareLaunchArgument(
            'rawOdomTopic',
            default_value='/odom'
        ),

        DeclareLaunchArgument(
            'mpcOdomTopic',
            default_value='/gmapping/odometry'
        ),

        DeclareLaunchArgument(
            'cmdTopic',
            default_value='/mpc_vel'
        ),

        DeclareLaunchArgument(
            'param_file',
            default_value='zz_unicycle_model_mpcc1'
        ),

        DeclareLaunchArgument(
            'db_filename',
            default_value=[
                EnvironmentVariable('HOME'),
                '/dbs/rl_learning.db'
            ]
        ),

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false'
        ),

        DeclareLaunchArgument(
            'slam_params_file',
            default_value=os.path.join(
                pkg_share, 'params', 'mapper_params_online_async.yaml',
            )
        ),

        # --- Parameter files ---
        Node(
            package='mpcc',
            executable='pearl_server.py',
            name='pearl_server',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'param_file': param_file,
            }]
        ),

        # Node(
        #     package='plotjuggler',
        #     executable='plotjuggler',
        #     name='plotjuggler',
        #     arguments=[
        #         '-l', juggler_layout_path, 
        #     ],
        #     output='screen'
        # ),

        Node(
            package='mpcc',
            executable='mpcc_ros',
            name='mpcc',
            output='screen',
            parameters=[
                ParameterFile(
                    PathJoinSubstitution([
                        pkg_share,
                        'params',
                        PythonExpression(["'", param_file, "' + '.yaml'"]),
                    ]),
                    allow_substs=True
                ),
                PathJoinSubstitution([
                    pkg_share,
                    'params',
                    'robo_params_ros2.yaml'
                ])
            ],
            remappings=[
                ('/odometry/filtered', mpc_odom_topic),
                # ('/cmd_vel', cmd_topic),  # uncomment if needed
            ],
            # prefix=['lldb -o run -- '],
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, 'launch', 'online_async_launch.py')
            ),
            launch_arguments={
                'use_sim_time': use_sim_time,
                'slam_params_file': slam_params_file,
            }.items()
        ),

        Node(
            package='mpcc',
            executable='publish_pf_pose',
            name='publish_pf_pose',
            output='screen',
            remappings=[
                ('/odometry/filtered', raw_odom_topic),
            ]
        )
    ])
