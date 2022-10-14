#!/usr/bin/env python3
# coding: UTF-8 
# Created by Cmoon

import rospy
from std_msgs.msg import String
from soundplayer import Soundplayer
from pdfmaker import Pdfmaker

COMMAND = {
    'follow':['follow','Follow','Follow.']
}


class Recognizer:
    def __init__(self):
        rospy.Subscriber('/xfspeech', String, self.talkback)
        self.wakeup = rospy.Publisher('/xfwakeup', String, queue_size=10)
        self.start_signal = rospy.Publisher('/follow', String, queue_size=10)
        self.cmd = None
        self.command = COMMAND
        self.goal = ''
        self._soundplayer = Soundplayer()
        self._pdfmaker = Pdfmaker()
        self.status = 0
        self.key = 1
        self.get_cmd() 

    def talkback(self, msg):
        if self.key == 1:
            print("\n讯飞读入的信息为: " + msg.data)
            self.cmd = self.processed_cmd(msg.data)
            self.judge()

    def judge(self):
        print("start judge!")
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
            # self._pdfmaker.write('Cmd: Do you need me start ' + self.goal + ' ?')
            # self._pdfmaker.write('Respond: Ok,I will.')
            print('Ok, I will.')
            self.start_signal.publish(self.goal)
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
        for (key, val) in self.command.items():
            for word in val:
                if word in self.cmd:
                    self.goal = key
                    print("key:",self.key)
                    response = response + ' start ' + key + ' ?'
                    break
        return response


if __name__ == '__main__':
    try:
        rospy.init_node('voice_recognition')
        Recognizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
