#!/usr/bin/env python
# coding: UTF-8 

import rospy
import math
from visualization_msgs.msg import MarkerArray
from soundplayer import Soundplayer
from geometry_msgs.msg import Twist
from base_controller import Base

JOINT = ['pelvis', 'navel', 'chest', 'neck', 'left clavicle', 'left shoulder', 'left elbow', 'left wrist', 'left hand',
         'left handtip', 'left thumb', 'right clavicle', 'right shoulder', 'right elbow', 'right wrist', 'right hand',
         'right handtip', 'right thumb', 'left hip', 'left knee', 'left ankle', 'left foot', 'right hip', 'right knee',
         'right ankle', 'right foot', 'head', 'nose', 'left eye', 'left ear', 'right eye', 'right ear', 'count']


class Findbody:
    def __init__(self):
        self.soundplayer = Soundplayer()
        self.base = Base()
        self.id0 = None
        self.id = None
        self.joint = None
        self.px = None
        self.py = None
        self.pz = None
        self.ox = None
        self.oy = None
        self.oz = None
        self.ow = None
        self.flag = True
        self.lock = False
        self.kp_go = 4.0
        self.td_go = 1.0
        self.kp_turn = 1.0
        self.aim = 0.7
        self.error_go = 1.5
        self.key = 0
        rospy.Subscriber('/k4a/body_tracking_data', MarkerArray, self.save)
        self.speed = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    def save(self, msg):
        if len(msg.markers) == 0 and self.id0 is None:
            print('finding')
            self.base.rotate(0.4)
        elif len(msg.markers) >= 3:
            if self.flag:
                self.id0 = msg.markers[2].id / 100
                self.base.stop()
                rospy.loginfo('I have found a person.')
                self.soundplayer.say('I have found a person.')
                self.flag = not self.flag

            self.id = msg.markers[2].id / 100
            self.joint = msg.markers[2].id % 100
            if self.id == self.id0:
                self.px = msg.markers[2].pose.position.x
                self.py = msg.markers[2].pose.position.y
                self.pz = msg.markers[2].pose.position.z
                self.ox = msg.markers[2].pose.orientation.x
                self.oy = msg.markers[2].pose.orientation.y
                self.oz = msg.markers[2].pose.orientation.z
                self.ow = msg.markers[2].pose.orientation.w
                last_error = self.pz if self.key == 0 else self.error_go
                self.error_go = self.pz - self.aim
                error_turn = self.px
                rospy.loginfo('[{}],[{}],[{}]'.format(self.pz, self.px, self.error_go))
                cmd = Twist()
                cmd.linear.x = 0.3 if self.kp_go * self.error_go > 0.6 else self.kp_go * self.error_go + self.td_go * (
                        self.error_go - last_error)
                cmd.angular.z = -1 * self.kp_turn * error_turn if (math.fabs(error_turn) > 0.2) else 0
                cmd.angular.z = -1 * self.kp_turn * error_turn if (math.fabs(error_turn) > 0.2) else 0
                self.speed.publish(cmd)
            else:
                self.speed.publish(Twist())


if __name__ == '__main__':
    try:
        rospy.init_node('findbody', anonymous=True)
        finder = Findbody()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
