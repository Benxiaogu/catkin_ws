#!/usr/bin/python
# coding: utf-8
# Created by Cmoon

import math
import rospy
import sys
from nav_msgs.msg import Odometry  # Pose数据类型包含机器人的坐标和角度
from geometry_msgs.msg import Twist  # Twist数据类型包含线速度和角速度
from tf import transformations as ts


class Turtle:
    def __init__(self, name, graph):
        rospy.init_node(name)  # 初始化节点
        rospy.Subscriber('/odom', Odometry, self.control)  # 实例化订阅者，参数为订阅的话题名，消息类型，回调函数
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)  # 实例化发布者，参数为发布的话题名，消息类型，队列长度
        self._graph = graph
        self.size = 2  # 图形的大小(3-5)
        self.kp1 = 0.5  # 走直线的比例控制参数
        self.kp2 = 1  # 转角度的比例控制参数
        self.kd = 0.5  # 走直线的微分控制参数
        self.aim_line = 0.03  # 直线的误差值
        self.aim_rotate = 0.01  # 转弯的误差值
        self.vel_cmd = Twist()  # 实例化Twist消息类型的消息
        self.point = {}  # 储存所走路径目标点
        self.x = None  # 机器人目前所在x坐标
        self.y = None  # 机器人目前所在y坐标
        self.theta = None  # 机器人目前角度
        self.goal = 0  # 机器人下一个要到达的目标点
        self.error = None  # 现在距离目标点的误差值
        self.quaternion = None  # 机器人的四元数
        self.euler = None  # 机器人的欧拉角
        self.flag = 0  # flag和lock实现走直线和转角度的互锁，执行完一个才能执行另一个
        self.lock = 0
        self.key = 0

    def control(self, pose):
        """订阅的回调函数"""
        self.choose_graph(self._graph, pose)  # 设定要走什么形状

    def choose_graph(self, graph, pose):
        """根据传入的图形设定目标点，获取当前位置，控制运动"""
        self.set_goal_points(pose, self.size)
        rospy.loginfo(self.point['squ'][0][0])
        self.get_present_point(pose)
        self.go_graph(graph, self.kp1, self.kp2, self.kd, self.aim_line)

    def set_goal_points(self, pose, size=5):
        """设定不同形状的目标点，坐标和角度"""
        if self.key == 0:  # 只设定一次
            self.point = {'squ': [[pose.pose.pose.position.x, pose.pose.pose.position.y, 0],
                                  [pose.pose.pose.position.x + size, pose.pose.pose.position.y, math.pi / 2],
                                  [pose.pose.pose.position.x + size, pose.pose.pose.position.y + size, math.pi],
                                  [pose.pose.pose.position.x, pose.pose.pose.position.y + size, - math.pi / 2]
                                  ],

                          'rec': [[pose.pose.pose.position.x, pose.pose.pose.position.y, 0],
                                  [pose.pose.pose.position.x + 4, pose.pose.pose.position.y, 0 + math.pi / 2],
                                  [pose.pose.pose.position.x + 4, pose.pose.pose.position.y + 3, 0 + math.pi],
                                  [pose.pose.pose.position.x, pose.pose.pose.position.y + 3, 0 - math.pi / 2]
                                  ],

                          'tri_60': [[pose.pose.pose.position.x, pose.pose.pose.position.y, 0],
                                     [pose.pose.pose.position.x + size, pose.pose.pose.position.y, math.pi * 2 / 3],
                                     [pose.pose.pose.position.x + size / 2,
                                      pose.pose.pose.position.y + (size / 2 * math.tan(math.pi / 3)),
                                      0 - math.pi * 2 / 3]
                                     ],

                          'tri_90': [[pose.pose.pose.position.x, pose.pose.pose.position.y, 0],
                                     [pose.pose.pose.position.x + size, pose.pose.pose.position.y, math.pi / 2],
                                     [pose.pose.pose.position.x + size, pose.pose.pose.position.y + size,
                                      -3 * math.pi / 4],
                                     ],
                          }
            self.key += 1

    def get_present_point(self, pose):
        """获取当前坐标和角度"""
        self.x = pose.pose.pose.position.x
        self.y = pose.pose.pose.position.y
        self.quaternion = [pose.pose.pose.orientation.x, pose.pose.pose.orientation.y, pose.pose.pose.orientation.z,
                           pose.pose.pose.orientation.w]
        self.euler = ts.euler_from_quaternion(self.quaternion)
        self.theta = self.euler[2]

    def go_graph(self, graph, kp1=4.0, kp2=4.0, kd=15.0, aim=0.025):
        """控制运动"""
        self.go_line(graph, kp1, kd, aim)
        self.rotate(graph, kp2)

    def set_goal(self, graph):
        """设定下一个目标点"""
        if self.lock == 1:
            print(self.goal)
            self.goal = self.goal + 1 if self.goal != len(self.point[graph]) - 1 else 0

    def go_line(self, graph, kp=2.0, kd=15.0, aim=0.08, ):
        """控制走直线"""
        if self.lock == 0:  # 和旋转互锁
            if self.flag == 0:  # 第一次进函数执行一次初始化
                self.error = self.size  # 初始误差为设定的图形大小
                self.flag += 1  # 保证初始化只执行一次
            last_error = self.error  # 上一次的误差
            self.error = math.sqrt(
                (self.x - self.point[graph][self.goal][0]) ** 2 + (
                        self.y - self.point[graph][self.goal][1]) ** 2)  # 计算现在的误差
            if abs(self.error) > aim:  # 误差大于设定精度时前进
                self.vel_cmd.linear.x = kp * (self.error + 0.1) + kd * (self.error - last_error)  # pd控制计算当前速度
                rospy.loginfo('mode:going')
                rospy.loginfo('goal:{},error:{},speed:{}'.format(self.goal, self.error, self.vel_cmd.linear.x))
            else:
                self.vel_cmd.linear.x = 0
                self.lock = 1  # 解锁旋转
            self.pub.publish(self.vel_cmd)  # 发布机器人速度

    def rotate(self, graph, kp=2.0):
        """控制旋转"""
        if self.lock == 1:  # 和直走互锁
            if self.flag == 1:  # 第一次进函数执行一次初始化函数
                self.set_goal(graph)
                theta1 = self.theta + 2 * math.pi if self.theta < 0 else self.theta  # 将现在的角度换算成0-2π
                theta2 = theta1 + math.pi if theta1 <= math.pi else theta1 - math.pi  # 和现在的角度差π的角
                target = self.point[graph][self.goal - 1][2] + 2 * math.pi if self.point[graph][self.goal - 1][2] < 0 \
                    else self.point[graph][self.goal - 1][2]  # 将目标角度换算到0-2π
                self.speed = kp if theta1 < target < theta2 or theta1 > theta2 and not (theta1 > target > theta2) \
                    else -kp  # 判断顺时针转还是逆时针转距离最短
                self.flag += 1  # 保证只执行一次初始化
            self.error = abs(self.theta - self.point[graph][self.goal - 1][2])  # 计算当前误差
            if self.error > self.aim_rotate:  # 误差大于0.01时保持旋转
                self.vel_cmd.angular.z = kp * (self.error + 0.1)  # p控制计算旋转速度
                rospy.loginfo('mode:rotating')
                rospy.loginfo('goal:{},error:{},speed:{}'.format(self.goal, self.error, self.vel_cmd.angular.z))
            else:
                self.vel_cmd.angular.z = 0
                self.flag = 0
                self.lock = 0
            self.pub.publish(self.vel_cmd)  # 发布机器人速度


if __name__ == '__main__':
    graph_list = ['squ', 'rec', 'tri_60', 'tri_90']
    if len(sys.argv) > 1 and sys.argv[1] in graph_list:  # 实现rosrun xx yy.py squ直接跑对应图形
        command = sys.argv[1]
    else:
        command = raw_input('Please input graph name(squ tri_60 tri_90 rec): ')  # 没在命令输对图形名称时提示输入
    try:
        Turtle('turtle_graph', command)  # 实例化Turtle，传入初始化的节点名s
        rospy.spin()  # 循环监听callback
    except rospy.ROSInterruptException:
        rospy.loginfo("Keyboard interrupt.")
