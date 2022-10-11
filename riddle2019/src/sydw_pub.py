#!/usr/bin/env python
# coding:utf-8
'''sydw ROS Node'''
# license removed for brevity
# 生源定位，灵犀灵犀唤醒词
import rospy
import os
import sys
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
from std_msgs.msg import Int16
import math
import time
import thread

import logging
logging.basicConfig()

cnt = 0
wakeupcount = 0
class sydw:
    def __init__(self, script_path):
        rospy.init_node("sydw")
        self.pub=rospy.Publisher('/cmd_vel',Twist,queue_size=10)
        self.pub2 = rospy.Publisher('/xfwakeup',String,queue_size=10)
        self.sub=rospy.Subscriber('/Shengyuan',Int16,self.abs)
        self.angle = 0
        self.lock = False
        self.location = 0
        self.start = True
    def abs(self,data):
        global wakeupcount
        if(wakeupcount < 5):                    #第一阶段
            msg1 = String()
            msg1.data = "ok"
            rospy.loginfo("speak out")
            self.pub2.publish(msg1)
            wakeupcount += 1
        else:                               #第二阶段
            thread.start_new_thread(self.adjust,(data,))
        print(wakeupcount)
        #rospy.loginfo(float(data.data))
    def adjust(self,data):
        #rospy.loginfo("get")
        #rospy.loginfo(data.data)
        vel_attitude = -1.0
        try:
            get_angle = float(data.data)
        except:
            return 
        if(get_angle > 180):
            get_angle = 360 - get_angle
            vel_attitude = 1.0
        if(get_angle == self.location):
            return 
        self.location = get_angle
        #rospy.loginfo(self.start)
        if(get_angle>0):
            fabs = get_angle
        else:
            fabs = -get_angle
        rate = rospy.Rate(3)
        if(self.lock == True):
            self.lock = False
        else:
            self.lock = True
        sublock = self.lock
        global cnt
        cnt += 1
        jiasu = self.start
        self.start = False
        flag = False
        while(self.lock == sublock):    #angle
            msg = Twist()
            if(get_angle > self.angle):
                value = 12
            elif(get_angle < self.angle):
                value = -12
            else:
                value = 0
            #msg.angular.z=float(value)
            msg.angular.z=float(value)*(vel_attitude)
            if(jiasu == True):    #kaolvjiasu
                if(self.angle < 20):
                    value = 2.5*math.sqrt(self.angle)+1
            self.angle += value
            self.pub.publish(msg)
            rate.sleep()
            rospy.loginfo(self.angle)
            if(math.fabs((self.angle - fabs))<=value):
                flag = True
                msg1 = String()
                msg1.data = "ok"
                rospy.loginfo("speak out")
                self.pub2.publish(msg1)
                break
        self.angle = 0
        self.start = flag
        
if __name__=="__main__":
    try:
        sydw(sys.path[0])
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("sydw class has not been constructed. Something is error.")
