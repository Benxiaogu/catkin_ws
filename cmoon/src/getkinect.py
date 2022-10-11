#!/usr/bin/env python3
# coding: UTF-8


import rospy
import cv2
from aip import AipBodyAnalysis, AipFace
import base64
import os
import numpy as np
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.augmentations import letterbox
from utils.plots import Annotator, colors
from utils.torch_utils import time_sync
from pyKinectAzure import pyKinectAzure, _k4a

from Detector import FaceDetector,BodyDetector,ObjectDetector 
from sensor_msgs.msg import Image
from std_msgs.msg import String
    


if __name__ == '__main__':
    try:
        rospy.init_node('name', anonymous=True)

        yolo = ObjectDetector()
        # name, result = yolo.detect(device='k4a', mode='realtime', find=None, depth=False, rotate=True)
        # print(name[0])
        # print(result)
        # name = yolo.detect(device='k4a', mode='realtime', attributes=None, depth=True)
        # print('name:{}'.format(name))

        result = rospy.Subscriber('/k4a/rgb/image_raw', Image, yolo.detect(device='k4a'), queue_size=1, buff_size=52428800)#使用kinect实时检测
        # rospy.Subscriber("/usb_cam/image_raw", Image, self.changeform, queue_size=1, buff_size=52428800)  # 使用电脑摄像头实时检测
        pub = rospy.Publisher('/rgb_image', Image, queue_size=10)
        pub.publish(result)
    except rospy.ROSInterruptException:
        pass
    
