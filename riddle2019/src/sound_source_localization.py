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
from nav_msgs.msg import Odometry
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
        self.sub2=rospy.Subscriber('/odom',Odometry,self.get_imu_angle)
        time.sleep(1)#获取位置需要时间
        self.sub=rospy.Subscriber('/Shengyuan',Int16,self.abs)
        self.angle = 0
    def get_imu_angle(self, odom_msg):
        w = odom_msg.pose.pose.orientation.w
        z = odom_msg.pose.pose.orientation.z
        angle = math.atan2(2*w*z,1-2*z*z)
        
        #rospy.loginfo(angle_msg)
        x = math.atan2(2,6)
        self.angle = angle*180/3.14+180
        #print(self.angle)
        #rospy.loginfo(self.angle)
    def abs(self,data):
        global wakeupcount
        if(wakeupcount < 10000):                    #第一阶段
            msg1 = String()
            msg1.data = "ok"
            rospy.loginfo("speak out")
            self.pub2.publish(msg1)
            wakeupcount += 1
        else:                               #第二阶段
            msg1 = String()
            msg1.data = "ok"
            rospy.loginfo("speak out")
            self.pub2.publish(msg1)
            thread.start_new_thread(self.adjust,(data,))
        print(wakeupcount)
        #rospy.loginfo(float(data.data))
    def adjust(self,data):
        if(data.data > 180):
            angle_change = 360-data.data
        else:
            angle_change = data.data
        #print("angle change",angle_change)
        angle_start = self.angle               #初始角度
        #print("start angle",self.angle )
        if(data.data > 180):
            speedpoint = 1
        else:
            speedpoint = -1
        angle_end = (angle_start + speedpoint*angle_change)%360
        if(angle_end < 0):
            angle_end += 360
        rate = rospy.Rate(10)
        msg = Twist()
        #print("end angle",angle_end)
        while(math.fabs(self.angle - angle_end)>5):
            if(math.fabs(self.angle - angle_end) < 20):            #开始刹车
                msg.angular.z = 0.2*speedpoint
            else:
                msg.angular.z = 0.8*speedpoint
            # print("position",self.angle)
            self.pub.publish(msg)
            rate.sleep()
        msg.angular.z = 0
        self.pub.publish(msg)
        #msg1 = String()
        #msg1.data = "ok"
        rospy.loginfo("speak out")
        #self.pub2.publish(msg1)
        #print("end_position",self.angle)

        
if __name__=="__main__":
    try:
        sydw(sys.path[0])
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("sydw class has not been constructed. Something is error.")
