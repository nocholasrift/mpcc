import os
import sys
import time
import yaml
import torch
import rospy
import signal
import rospkg
import argparse
import subprocess
import numpy as np

from tqdm import tqdm
from os.path import join
from simple_simulation import SimpleSimulation

INIT_POSITION = [-2.25, -2, 1.57]  # in world frame
# INIT_POSITION = [-2.25, 8, -1.57]  # in world frame
GOAL_POSITION = [0, 10]  # relative to the initial position

bag_process = None
gazebo_process = None
controller_process = None
planner_process = None

bag_dir = None


def compute_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def shutdown():
    global gazebo_process, controller_process, planner_process

    rospy.loginfo("*************** shutting nodes down now! ***************")
    if bag_process is not None:
        bag_process.terminate()
        bag_process.wait()

    if gazebo_process is not None:
        gazebo_process.terminate()
        gazebo_process.wait()

    if controller_process is not None:
        controller_process.terminate()
        controller_process.wait()

    if planner_process is not None:
        planner_process.terminate()
        planner_process.wait()


def myexcepthook(type, value, tb):
    global mvsim_process, controller_process, planner_process, bag_process

    print("*************** shutting nodes down now! ***************")
    if bag_process is not None:
        bag_process.terminate()
        bag_process.wait()

    if gazebo_process is not None:
        gazebo_process.terminate()
        gazebo_process.wait()

    if controller_process is not None:
        controller_process.terminate()
        controller_process.wait()

    if planner_process is not None:
        planner_process.terminate()
        planner_process.wait()

    # print the error
    print(type)
    print(value)


def path_coord_to_gazebo_coord(x, y):
    RADIUS = 0.075
    r_shift = -RADIUS - (30 * RADIUS * 2)
    c_shift = RADIUS + 5

    gazebo_x = x * (RADIUS * 2) + r_shift
    gazebo_y = y * (RADIUS * 2) + c_shift

    return (gazebo_x, gazebo_y)


def run_sim(args):
    global gazebo_process, controller_process, planner_process, bag_process, INIT_POSITION, GOAL_POSITION

    rospack = rospkg.RosPack()
    base_path = rospack.get_path("mpcc")
    planner_path = rospack.get_path("robust_fast_navigation")
    world_path = rospack.get_path("jackal_helper")
    sim_path = rospack.get_path("cbf_tracking")
    world_name = f"BARN/world_{args.world_idx}.world"

    print(
        ">>>>>>>>>>>>>>>>>> Loading Simulation with %s <<<<<<<<<<<<<<<<<<"
        % (world_name)
    )

    if args.double_integrator:
        launch_file = join(sim_path, "launch", "double_integrator.launch")
    else:
        launch_file = join(sim_path, "launch", "simulator.launch")

    world_name = join(world_path, "worlds", world_name)

    gazebo_process = subprocess.Popen(
        [
            "roslaunch",
            launch_file,
            "world_name:=" + world_name,
        ]
    )
    time.sleep(5)

    collision_topic = "/mpc_done" if args.train else "/collision"
    simple_sim = SimpleSimulation(collision_topic)

    init_coor = (INIT_POSITION[0], INIT_POSITION[1])
    # due to map / world transform, flip goal_pos coords...
    goal_coor = (
        INIT_POSITION[0] + GOAL_POSITION[0],
        INIT_POSITION[1] + GOAL_POSITION[1],
    )

    pos = simple_sim.get_model_state().odom.pose.pose.position
    curr_coor = (pos.x, pos.y)
    collided = True

    # check whether the robot is reset, the collision is False
    while compute_distance(init_coor, curr_coor) > 0.1 or collided:
        pos = simple_sim.get_model_state().odom.pose.pose.position
        print(pos)
        curr_coor = (pos.x, pos.y)
        collided = simple_sim.get_hard_collision()
        time.sleep(1)

    bag_fname = None
    bag_dir = None

    if args.bag:

        from datetime import datetime

        bag_dir = rospy.get_param("/train/bag_dir", "./")
        dt_str = "{:%Y_%m_%d-%H-%M-%S}".format(datetime.now())
        bag_fname = f"world_{args.world_idx}_{dt_str}.bag"
        bag_launch_file = os.path.join(base_path, "launch/record_bag.launch")

        if not os.path.exists(bag_dir):
            os.makedirs(bag_dir)

        print("opening bag file at", (os.path.join(bag_dir, bag_fname)))

        bag_process = subprocess.Popen(
            [
                "roslaunch",
                bag_launch_file,
                "path:=" + bag_dir,
                "bag_name:=" + bag_fname,
            ],
            # executable="/usr/bin/zsh",
        )

    time.sleep(5)

    nav_stack_launch_file = join(base_path, "launch/jackal_mpc_track.launch")
    # if args.double_integrator:
    #     nav_stack_launch_file = join(base_path, "launch/jackal_di_track.launch")

    params = ""
    if args.eval:
        params = ("use_rl:=true",)

    controller_process = subprocess.Popen(
        ["roslaunch", nav_stack_launch_file, *params],
        # executable="/usr/bin/zsh",
    )
    time.sleep(3)

    planner_launch_file = join(planner_path, "launch/planner_gurobi.launch")
    planner_process = subprocess.Popen(
        [
            "roslaunch",
            planner_launch_file,
            "barn:=true",
            "barn_dist:=10.0",
        ],
        # executable="/usr/bin/zsh",
    )

    ##########################################################################
    # 2. Start navigation
    ##########################################################################

    curr_time = rospy.get_time()
    pos = simple_sim.get_model_state().odom.pose.pose.position
    curr_coor = (pos.x, pos.y)

    prog_crash = False
    start_time = curr_time
    # check whether the robot started to move
    while (
        compute_distance(init_coor, curr_coor) < 0.1
        or planner_process.poll() is not None
    ):

        if curr_time - start_time > 60:
            prog_crash = True
            break

        curr_time = rospy.get_time()
        pos = simple_sim.get_model_state().odom.pose.pose.position
        curr_coor = (pos.x, pos.y)
        time.sleep(0.01)

    # start navigation, check position, time and collision
    start_time = curr_time
    collided = False
    timeout_time = 50

    if planner_process.poll() is not None or prog_crash:
        collided = True

    while (
        compute_distance(goal_coor, curr_coor) > 1
        and not collided
        and curr_time - start_time < timeout_time
    ):
        curr_time = rospy.get_time()
        pos = simple_sim.get_model_state().odom.pose.pose.position

        curr_coor = (pos.x, pos.y)
        print(
            "Time: %.2f (s), x: %.2f (m), y: %.2f (m)"
            % (curr_time - start_time, *curr_coor),
            end="\r",
        )
        collided = simple_sim.get_hard_collision()
        while rospy.get_time() - curr_time < 0.1:
            time.sleep(0.01)

    print(">>>>>>>>>>>>>>>>>> Test finished! <<<<<<<<<<<<<<<<<<")

    success = False
    if collided:
        status = "collided"
    elif curr_time - start_time >= timeout_time:
        status = "timeout"
    else:
        status = "succeeded"
        success = True
    print("Navigation %s with time %.4f (s)" % (status, curr_time - start_time))

    nav_metric = 1 if success else 0
    print("Navigation metric: %.4f" % (nav_metric))

    with open(args.out, "a") as f:
        f.write(
            "%d %d %d %d %.4f %.4f\n"
            % (
                args.world_idx,
                success,
                collided,
                (curr_time - start_time) >= timeout_time,
                curr_time - start_time,
                nav_metric,
            )
        )

    if bag_process is not None:
        bag_process.terminate()
        bag_process.wait()

    if args.bag and status == "timeout":
        print(
            "Navigation timed out, so deleting bag file",
            os.path.join(bag_dir, bag_fname),
        )
        os.remove(os.path.join(bag_dir, bag_fname))

    gazebo_process.terminate()
    gazebo_process.wait()
    controller_process.terminate()
    controller_process.wait()
    planner_process.terminate()
    planner_process.wait()

    time.sleep(2)

    return success


if __name__ == "__main__":

    original_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, shutdown)
    sys.excepthook = myexcepthook

    parser = argparse.ArgumentParser(description="test BARN navigation challenge")
    parser.add_argument("--world_idx", type=int, default=-1)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--out", type=str, default="out.txt")
    parser.add_argument("--bag", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test_suite", action="store_true")
    parser.add_argument("--double_integrator", action="store_true")
    args = parser.parse_args()

    if args.train and args.eval:
        print("Cannot train and evaluate at the same time")
        sys.exit(-1)

    rospy.init_node("gym", anonymous=True)  # , log_level=rospy.FATAL)
    # rospy.set_param("/use_sim_time", True)
    rospy.on_shutdown(shutdown)

    # load parameter from yaml file
    rospack = rospkg.RosPack()
    base_path = rospack.get_path("mpcc")

    # test writing to output file to make sure it works
    with open(args.out, "w") as f:
        f.write("Below are the simulation results for the test trials\n")

    trainer = None

    rospy.loginfo("Starting simulation")
    eval_worlds = np.array(
        [
            10,
            14,
            17,
            38,
            44,
            87,
            90,
            91,
            92,
            93,
            95,
            101,
            105,
            108,
            109,
            115,
            116,
            128,
            130,
            133,
            135,
            136,
            139,
            144,
            145,
            146,
            147,
            149,
            151,
            152,
            160,
            161,
            178,
            187,
            189,
            190,
            193,
            236,
            245,
            247,
            249,
            251,
            252,
            263,
            267,
            270,
            274,
            275,
            278,
            290,
        ]
    )

    world_list = np.array([i for i in range(0, 300)])

    # print(world_list)

    if args.world_idx >= 0:
        count = 0
        for i in range(1, 40):
            success = run_sim(args)

            if success == 1:
                count += 1
            elif success == 0:
                count = 0
            else:
                print("simulation error occured on trial", i)
    else:
        # sample 50 worlds out of 300 excluding 200-229
        eval_worlds = np.sort(eval_worlds)
        for world in eval_worlds:

            for i in range(5):
                args.world_idx = world
                run_sim(args)

    rospy.signal_shutdown("finished")
