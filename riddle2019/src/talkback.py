#!/usr/bin/env python
# coding:utf-8
'''talk back Node'''
import rospy
import sys
from std_msgs.msg import String
from sound_play.libsoundplay import SoundClient

import logging
logging.basicConfig()

class Talk:      #关键词匹配
    def __init__(self, script_path):
        rospy.init_node("talkback")
        self.sub = rospy.Subscriber('/xfwords', String, self.talkback)
        self.soundhandle = SoundClient()
        rospy.sleep(1)
    def talkback(self, text):
        self.soundhandle.say(text.data)

if __name__ == "__main__":
    try:
        Talk(sys.path[0])
        rospy.spin()
    except Exception as e:
        rospy.loginfo("讲述节点无法打开")
        print(e)
