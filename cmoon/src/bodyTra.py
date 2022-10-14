#! /usr/bin/python3
# coding: UTF-8


import rospy
import cv2
import sys
sys.path.insert(1, '../')

from std_msgs.msg import String


class BodyFinding:
    def __init__(self):
        rospy.Subscriber('/k4a/')
        
