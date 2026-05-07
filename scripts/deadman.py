#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

class DeadmanSwitch:
    def __init__(self):
        rospy.init_node('deadman_switch_node')

        # --- Parameters ---
        self.deadman_button_index = 7
        self.is_pressed = False

        # --- Publishers & Subscribers ---
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        self.mpc_msg = None
        self.teleop_msg = None

        self.zero_msg = Twist()
        self.zero_msg.linear.x = 0
        self.zero_msg.angular.z = 0
        
        rospy.Subscriber('/joy', Joy, self.joy_callback)
        rospy.Subscriber('/teleop_cmd_vel', Twist, self.twist_callback)
        rospy.Subscriber('/mpc_vel', Twist, self.mpc_callback, queue_size=1)
        timer = rospy.Timer(rospy.Duration(0.01), self.ctrl_loop)

    def joy_callback(self, msg):
        self.is_pressed = bool(msg.buttons[self.deadman_button_index])

    def twist_callback(self, msg):
        self.teleop_msg = msg

    def mpc_callback(self, msg):
        self.mpc_msg = msg

    def ctrl_loop(self, event):

        if self.is_pressed and self.mpc_msg != None:
            self.pub.publish(self.mpc_msg)
        elif self.teleop_msg != None:
            self.pub.publish(self.teleop_msg)
        else:
            self.pub.publish(self.zero_msg)
        

if __name__ == '__main__':
    node = DeadmanSwitch()
    rospy.spin()
