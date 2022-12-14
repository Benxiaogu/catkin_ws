#!/usr/bin/env python3
# coding: UTF-8
# Created by Cmoon

import rospy
from navigator import Navigator  # 导航模块
from soundplayer import Soundplayer  # 语音合成模块
from voice_recognizer import Recognizer  # 语音识别模块和分析模块
from pdfmaker import Pdfmaker  # pdf制作模块
from base_controller import Base  # 底盘运动模块
from std_msgs.msg import String  # String类型消息,从String.data中可获得信息
from Detector import ObjectDetector

# LOCATION = {  # 储存导航路径点
#     'door': [[-4.352973, -6.186659, 0.000000], [0.000000, 0.000000, -0.202218, -0.979341]],
#     'living room': [[-0.476640, -4.946882, 0.000000], [0.000000, 0.000000, 0.808888, 0.587962]],
#     'kitchen': [[-1.658400, -0.046712, 0.000000], [0.000000, 0.000000, -0.986665, 0.162761]],
#     'bedroom': [[3.859466, -2.201285, 0.000000], [0.000000, 0.000000, -0.247601, -0.968862]],
#     'dining room': [[3.583689, 0.334696, 0.000000], [0.000000, 0.000000, -0.820933, -0.571025]],
#     'garage': [[0.166213, 3.886673, 0.000000], [0.000000, 0.000000, -0.982742, 0.184983]],
#     'rubbishbin1': [[-3.026017, -5.607293, 0.000000], [0.000000, 0.000000, -0.569564, 0.821947]],
#     'rubbishbin2': [[-1.275007, 0.378040, 0.000000], [0.000000, 0.000000, -0.817479, -0.575959]],
#     'rubbishbin3': [[3.978401, 3.763957, 0.000000], [0.000000, 0.000000, -0.213188, -0.977011]]
# }

RUBBISH = {
    'rubbishbin1': ['bottle', 'cup'],
    'rubbishbin2': ['mouse'],
    'rubbishbin3': ['cell phone']
}



LOCATION = {
    'door':[[-0.611453, -0.0370463, 0.000000], [0.000000, 0.000000, -0.981173, 0.193130, 0.000000]],
    'living room': [[-1.138477, 2.171236, 0.000000], [0.000000, 0.000000, -0.905918, 0.423452]],
    'bedroom':[[-4.277096, 2.016742, 0.000000], [0.000000, 0.000000, 0.924435, 0.381339]],
    'kitchen':[[-4.415931, -0.193947, 0.000000], [0.000000, 0.000000, -0.999249, 0.038723]]
}



class Controller:
    def __init__(self, name):
        rospy.init_node(name, anonymous=True)  # 初始化ros节点
        rospy.Subscriber('/start_signal', String, self.control)  # 创建订阅者订阅recognizer发出的地点作为启动信号
        # rospy.Subscriber('/yolo_result', String, self.save_result, queue_size=1)
        self.startyolo = rospy.Publisher('/ros2yolo', String, queue_size=1)
        self.navigator = Navigator(LOCATION)  # 实例化导航模块
        self.soundplayer = Soundplayer()  # 实例化语音合成模块
        self.recognizer = Recognizer()  # 实例化语音识别和逻辑判断模块
        self.pdfmaker = Pdfmaker()  # 实例化pdf导出模块
        self.base = Base()  # 实例化移动底盘模块
        self.weights = r'/home/sundawn/catkin_ws/src/cmoon/src/weights/yolov7.pt'
        self.yolo = ObjectDetector(weights=self.weights)
        self.soundplayer.say("Please give me the command.", 3)  # 语音合成模块调用play方法传入字符串即可播放
        self.recognizer.get_cmd()  # 获取一次语音命令
        self.result = None  # yolo检测的结果
        self.goal = None  # 要去清理垃圾的房间
        self.type = None  # 垃圾类型
        self.t = 0


    def control(self, place):
        """订阅start signal的回调函数,传入的place是String类型消息 .data可以获取传来的信息,即目标房间"""
        self.goal = place.data  # 存入目标房间名字
        for i in range(5):  # 循环五次
            self.navigator.goto(place.data)  # 导航模块调用goto方法,传入去的地点名字符串即可导航区指定地点
            self.detect()  # 检测垃圾
            self.catch()  # 抓取垃圾
            self.throw()  # 扔垃圾
            print(str(i + 1) + ' rubbish cleaned.')
        print('Task completed!')
        self.soundplayer.say('Task completed.')  # result的回调函数,实时更新检测到的物品名"""

    def detect(self):
        """启动yolo实时检测"""
        self.soundplayer.say('Starting to track the rubbish.')
        # print('Starting to track the rubbish.')
        self.result = None  # 检测前初始化检测结果

        findclasses = "bottle,mouse,cell phone,cup"
        self.result = self.yolo.detect(device="k4a",mode="detect",classes=findclasses,rotate=0.5)
        print(self.result)
        num = len(self.result)
        print('Object num:',num)
        self.t = 0
        if self.t == 0:
            for i in range(num):
                self.dedect_result = self.result[i].name
                print(self.dedect_result)
                self.judge(self.result[i].name)
                self.judge(self.dedect_result)

    def judge(self, result):
        """判断检测到的垃圾种类"""
        for (key, val) in RUBBISH.items():
            if result in val:
                self.type = key
                print('Rubbish:', result)
                print('Rubbish type : {}'.format(self.type))

    def catch(self):
        """抓取垃圾部分"""
        self.soundplayer.say('Please hand me the ' + self.result[0].name, 3)
        self.soundplayer.say('I have caught the rubbish.')
        print('I have caught the rubbish.')

    def throw(self):
        """扔垃圾部分"""
        self.soundplayer.say('I will go to throw the rubbish.')
        self.typegoal = 'door'
        print(self.typegoal)
        self.navigator.goto(self.typegoal)
        print('Throwing the rubbish...')
        rospy.sleep(3)
        self.soundplayer.say('I have thrown the rubbish.')
        print('I have thrown the rubbish.')


if __name__ == '__main__':
    try:
        Controller('controller')  # 实例化Controller,参数为初始化ros节点使用到的名字
        rospy.spin()  # 保持监听订阅者订阅的话题
    except rospy.ROSInterruptException:
        pass
