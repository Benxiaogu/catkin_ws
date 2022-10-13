# catkin_ws
#### 1.环境：

Ubuntu18.04

ROS-Melodic

YOLOV7



#### 2.通过下面命令下载：

```
git clone https://github.com/Benxiaogu/catkin_ws.git
```

将这些功能包放到工作空间src目录下进行编译即可使用

#### 3.cmoon包

原版地址：

https://github.com/Cmoon-cyl/ros-module

https://github.com/Cmoon-cyl/Object-Detector-YoloV7



#### 4.调用cmoon中的Detector.py中的Detector有两种方法:

Detector.py用于视觉识别，可调用：

BodyDetector()：人体检测

FaceDetector()：人脸识别

ObjectDetector()：物体识别

调用BodyDetector和FaceDetector需要先下载baidu-aip：

```
pip3 install baidu-aip
```

```
		body = BodyDetector()
        result2 = body.detect(
            ['age', 'gender', 'upper_wear', 'upper_wear_texture', 'upper_wear_fg', 'upper_color',
             'lower_wear', 'lower_color', 'face_mask', 'glasses', 'headwear', 'bag'], device='cam')
        
        face = FaceDetector()
        result1 = face.detect(attributes=['age', 'gender', 'glasses', 'beauty', 'mask'], device='cam')
        
```

关于调用ObjectDetector:

```
(1) 通过main()来调用ObjectDetector
导入模块：from Detector import ObjectDetector
调用：self.result = ObjectDetector.main()

(2) 直接调用Detector
导入模块：from Detector import ObjectDetector
实例化：self.yolo = ObjectDetector()
调用：self.result = self.yolo.detect()
device: 使用的设备(azure kinect/电脑摄像头)
mode: “realtime”(实时检测,按q退出) / “find”(检测到指定物品在画面范围内退出) / other(检测到任意物体在范围内退出)
range: 画面中心多大范围内的检测结果被采用
nosave: 是否保存图片到本地
find: 需要寻找的物体名称;
classes: 哪些物体可以被检测到,字符串名称间用','分割(例:传"bottle,person"则只会检测到bottle和person)
return: List of YoloResult(每个YoloResult代表一个物品)
如：
findclasses = "bottle,mouse,cell phone,cup"  # 只有这些物品可以被检测到
self.result = self.yolo.detect(device="k4a", mode="other",range=0.5,nosave=True,classes=findclasses, rotate=0.5)
```

#### 5.主程序

controller.py

controller1.py

#### 6.进行仿真

```
1.roslaunch cmoon rviz_monitu.launch

2.roslaunch riddle2019 riddle.launch

3.rosrun cmoon controller1.py
```

(仿真时要将controller1.py中的LOCATION改为上面一个，下面一个时实验室跑实体时的数据，仿真要用上面的LOCATION)

