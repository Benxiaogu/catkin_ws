#!/usr/bin/env python3
# coding:utf-8
import rospy
import math
import time
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import String
from geometry_msgs.msg import Twist

flag = False
find = False
follow_id = 0
scale_x = 4
scale_z = 1
goal_z = 0.8
start = time.time()

def lock(msg):
	global flag,find
	flag = not flag
	find = False

def call(msg):
	global flag,follow_id,find,pub1,pub2,start
	global goal_z,scale_z,scale_x
#	global pub
	if(flag and msg.markers):
		x = msg.markers[2].pose.position.x
		z = msg.markers[2].pose.position.z
#			print(x)
#		print(follow_id,math.fabs(x),msg.markers[2].id)
		if((not find) and math.fabs(x)<0.1):		#只找一次
			follow_id = msg.markers[2].id
			find =True
		if(msg.markers[2].id == follow_id):
#			print(follow_id,x,z)
			start = time.time()
			cmd = Twist()
			cmd.linear.x = 0.25 if scale_z*(z- goal_z)>0.25 else scale_z*(z- goal_z)
			cmd.angular.z = -1*scale_x*x if(math.fabs(x)>0.2) else 0
			if(z>1.5):
				pub2.publish('please wait for me')
			pub1.publish(cmd)
			print(cmd)
		else:
			if(follow_id and time.time()-start>3):
				find = False
			if(follow_id and time.time()-start>0.5):
				pub1.publish(Twist())

if __name__ == '__main__':
	rospy.init_node('Follow')
	pub1 = rospy.Publisher('/cmd_vel',Twist,queue_size=10)
	pub2 = rospy.Publisher('/xfwords',String,queue_size=1)
	rospy.Subscriber('follow',String,lock)
	rospy.Subscriber('/k4a/body_tracking_data',MarkerArray,call)
	rospy.spin()
