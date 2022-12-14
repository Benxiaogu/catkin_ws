#!/usr/bin/env python
# coding: UTF-8 
# Created by Cmoon

import rospy
from std_msgs.msg import String
from soundplayer import Soundplayer
from pdfmaker import Pdfmaker

LOCATION = {
    'door': ['door', 'dog'],
    'kitchen': ['kitchen', 'page', 'station', 'location', 'teacher'],
    'living room': ['living', 'leaving', 'livingroom', 'living room'],
    'bedroom': ['bedroom', 'bad', 'bed room', 'bad room', 'bed', 'bathroom', 'beijing'],
    'hallway': ['hallway', 'whole way'],
    'dining room': ['dining', 'dying'],
    'garage': ['garage']
}

COMMAND = {
    'follow':['follow','Follow','Follow.']
}


class Recognizer:
    def __init__(self):
        rospy.Subscriber('/xfspeech', String, self.talkback)
        # rospy.Subscriber('/voiceWords', String)
        self.wakeup = rospy.Publisher('/xfwakeup', String, queue_size=10)
        # self.wakeup = rospy.Publisher('/voiceWakeup', String, queue_size=10)
        self.start_signal = rospy.Publisher('/start_signal', String, queue_size=10)
        self.start_follow = rospy.Publisher('/follow', String, queue_size=10)
        self.cmd = None
        self.command = COMMAND
        self.location = LOCATION
        self.goal = ''
        self._soundplayer = Soundplayer()
        self._pdfmaker = Pdfmaker()
        self.status = 0
        self.key = 1
        self.goal_signal = 0
        self.goal_follow = 0

    def talkback(self, msg):
        if self.key == 1:
            print("\n讯飞读入的信息为: " + msg.data)
            self.cmd = self.processed_cmd(msg.data)
            self.judge()

    def judge(self):
        if self.status == 0:

            response = self.analyze()

            if response == 'Do you need me':
                self._soundplayer.say("Please say the command again. ")
                self.get_cmd()
            else:
                self.status = 1
                print(response)
                self._soundplayer.say(response, 3)
                self._soundplayer.say("please say yes or no.", 1)
                print('Please say yes or no.')
                self.get_cmd()

        elif ('Yes.' in self.cmd) or ('yes' in self.cmd) or ('Yeah' in self.cmd) or ('yeah' in self.cmd) and (
                self.status == 1):

            self._soundplayer.say('Ok, I will.')
            self._pdfmaker.write('Cmd: Do you need me go to the ' + self.goal + ' and clean the rubbish there?')
            self._pdfmaker.write('Respond: Ok,I will.')
            print('Ok, I will.')
            if self.goal_signal == 1:
                self.start_signal.publish(self.goal)
            elif self.goal_follow == 1:
                self.start_follow.publish(self.goal)
            self.key = 0
            self.status = 0
            self.goal = ''


        elif ('No.' in self.cmd) or ('no' in self.cmd) or ('oh' in self.cmd) or ('know' in self.cmd) and (
                self.status == 1):
            self._soundplayer.say("Please say the command again. ")
            print("Please say the command again. ")
            self.status = 0
            self.goal = ''
            self.get_cmd()

        else:
            self._soundplayer.say("please say yes or no.")
            print('Please say yes or no.')
            self.get_cmd()

    def processed_cmd(self, cmd):
        cmd = cmd.lower()
        for i in " ,.;?":
            cmd = cmd.replace(i, ' ')
        return cmd

    def get_cmd(self):
        """获取一次命令"""
        self._soundplayer.play('Speak.')
        self.wakeup.publish('ok')

    def analyze(self):
        response = 'Do you need me'
        for (key, val) in self.location.items():
            for word in val:
                if word in self.cmd:
                    self.goal = key
                    response = response + ' go to the ' + key + ' and throw the rubbish there?'
                    self.goal_signal = 1
                    break
        for (key, val) in self.command.items():
            for word in val:
                if word in self.cmd:
                    self.goal = key
                    print("key:",self.key)
                    response = response + ' start to ' + key + ' ?'
                    self.goal_follow = 1
                    break
        return response


if __name__ == '__main__':
    try:
        rospy.init_node('voice_recognition')
        Recognizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
