#!/usr/bin/env python3
# coding: UTF-8
# Created by Cmoon

import rospy
import cv2
from aip import AipBodyAnalysis, AipFace
import base64
import os
import numpy as np
import torch
from typing import *
from pathlib import Path
from numpy import random
import argparse

from base_controller import Base

import sys
sys.path.insert(1, '../')
import pykinect_azure as pykinect

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized, TracedModel

import pykinect_azure as pykinect

cv2_image = TypeVar("image opened by cv2")


class Detector(object):
    def __init__(self):

        # self.photopath = os.path.dirname(os.path.dirname(__file__)) + '/photo'
        self.photopath = r'/home/sundawn/图片'
        # self.photopath = '/home/sundawn/catkin_ws/src/cmoon/photo'
        self.base = Base()

    def get_file_content(self, filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    def take_photo(self, device='camera'):
        """电脑摄像头拍照保存"""
        if device == 'k4a' or device == 'kinect':
            print("using k4a")


            # Initialize the library, if the library is not found, add the library path as argument
            pykinect.initialize_libraries()

            # Modify camera configuration
            device_config = pykinect.default_configuration
            device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
            # print(device_config)

            # Start device
            device = pykinect.start_device(config=device_config)
            cv2.namedWindow('Face',cv2.WINDOW_NORMAL)
            path = self.photopath + '/photo.jpg'
            while True:

                # Get capture
                capture = device.update()

                # Get the color image from the capture
                ret, color_image = capture.get_color_image()
                if not ret:
                    continue
                # resolution = [1280, 720]
                resolution = [1980, 1080]
                cv2.imwrite(path, color_image)
                if 'photo.jpg' in os.listdir(self.photopath):
                    capture.release_handle()
                    break
            
            device.stop_cameras()
            device.close()
            faceimg = cv2.imread(path)
            cv2.imshow("Face",faceimg)
            cv2.waitKey(5000)
            cv2.destroyAllWindows() 

        else:
            cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
            cap.open(0)
            flag, frame = cap.read()
            # print(self.photopath)
            path = self.photopath + '/photo.jpg'
            cv2.imwrite(path, frame)
            cap.release()
            
            cv2.namedWindow("Face", cv2.WINDOW_NORMAL)  # 建立图像对象
            faceimg = cv2.imread(path)
            cv2.imshow("Face",faceimg)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()

        return path

    def get_attr(self, *key):
        return

    def detect(self, attributes=None, device='camera', mode=None, *keys):
        """电脑摄像头拍照检测"""
        path = self.take_photo(device)
        print("saving path:",path)
        result = self.get_attr(path, attributes)
        return result


class BodyDetector(Detector):
    """https://ai.baidu.com/ai-doc/BODY/6k3cpymz1"""

    def __init__(self):
        self.app_id = '24949761'
        self.api_key = '02d2yaps6uEDgOGRzMqs9Ggo'
        self.secret_key = 'NBDdIGGZtrLvCoi6aMxZ1Qd2uELcLiGU'
        self.client = AipBodyAnalysis(self.app_id, self.api_key, self.secret_key)
        super(BodyDetector, self).__init__()

    def get_attr(self, path=None, attributes=None):
        """
        example:path=r'/home/cmoon/图片/photo.jpeg'
        attributes=['age','gender','glasses',...]
        return:result字典,用result[age]等获得结果
        """
        attr = ''
        for items in attributes:
            attr = attr + items + ','
        image = self.get_file_content(path)
        options = {}
        # options["type"] = "gender,age,glasses,upper_color,upper_wear"
        options["type"] = attr
        """ 带参数调用人体检测与属性识别 """
        outcome = self.client.bodyAttr(image, options)
        result = {}
        for items in attributes:
            result[items] = outcome['person_info'][0]['attributes'][items]['name']
        return result


class FaceDetector(Detector):
    """https://ai.baidu.com/ai-doc/FACE/ek37c1qiz"""

    def __init__(self):
        self.app_id = '24950812'
        self.api_key = 'F3431IOfGwWIItzAvXfAMUpR'
        self.secret_key = 'vLGzRyiQpAtPphgMDlE2vAdjixFsOoAE'
        self.client = AipFace(self.app_id, self.api_key, self.secret_key)
        super(FaceDetector, self).__init__()


    def get_file_content(self, filePath):
        with open(filePath, 'rb') as fp:
            base64_data = base64.b64encode(fp.read())  # 使用base64进行加密
            return str(base64_data, 'utf-8')

    def get_attr(self, path=None, attributes=None):
        """
        example:path=r'/home/cmoon/图片/photo.jpeg'
        attributes=['age','gender','glasses',...]
        return：result字典,用result[age]等获得结果
        """
        attr = ''
        for items in attributes:
            attr = attr + items + ','

        image = self.get_file_content(path)
        imageType = "BASE64"
        self.client.detect(image, imageType)

        """ 如果有可选参数 """
        options = {}
        options["face_field"] = attr
        options["max_face_num"] = 2
        options["face_type"] = "LIVE"
        options["liveness_control"] = "LOW"

        """ 带参数调用人脸检测 """
        outcome = self.client.detect(image, imageType, options)
        result = {}
        if outcome['error_code'] == 0:
            result = {}
            # result['location'] = outcome['result']['face_list'][0]['location']
            for items in attributes:
                result[items] = outcome['result']['face_list'][0][items]

        return result

class YoloResult:
    """
    Yolo检测结果数据类型

    :param name: 标签名称
    :param box: 矩形框左上右下xyxy坐标
    :param x: 矩形框中心点x坐标
    :param y: 矩形框中心点y坐标
    :param conf: 置信度
    :param distance: 中心点深度值（暂未接入）
    """

    def __init__(self, name: str, box: List[int], x: int, y: int, conf: float, distance: float = None):
        self.name = name
        self.box = box
        self.x = x
        self.y = y
        self.conf = conf
        self.distance = distance

    def __str__(self):
        return f'name:{self.name}; box:{self.box}; x:{self.x}; y:{self.y}; conf:{self.conf:.2f}'


class ObjectDetector:
    """
    Object Detection YoloV7

    :param weights: .pt模型路径
    :param imgsz: 图片大小
    :param conf_thres: 置信度阈值
    :param iou_thres: iou阈值
    """

    def __init__(self, weights: Path = Path("weights", "yolov7.pt"), imgsz: int = 640,
                 conf_thres: float = 0.25, iou_thres: float = 0.45):
        self.weights = weights
        self.photo_path = Path(Path.cwd(), "images")
        self.device = select_device()
        self.save_dir = Path(Path.cwd(), "results")
        self.half = self.device.type != 'cpu'
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = None
        self.stride = None
        self.names = None
        self.colors = None
        self.base = Base()
        # self.weights = r'/home/sundawn/catkin_ws/src/cmoon/src/weights/yolov7.pt'
        print(f"weights:{self.weights}\nUsing device:{self.device}")

    def load_model(self) -> torch.nn.Module:
        """加载模型"""
        model = attempt_load(self.weights, map_location=self.device)
        self.stride = int(model.stride.max())
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size
        # model = TracedModel(model, self.device, self.imgsz)
        if self.half:
            model.half()
            model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(model.parameters())))
        self.names = model.module.names if hasattr(model, 'module') else model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        return model

    def process_img(self, img0: cv2_image) -> np.ndarray:
        """
        图像预处理

        :param img0: opencv读取的原始图像
        :return: 处理过的图像,直接传入模型
        """
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def process_result(self, pred: list, img: cv2_image, img0: cv2_image, t0, t1, t2, t3) -> List[YoloResult]:
        """
        处理检测结果

        :param pred: 模型输出结果
        :param img: 经process_img处理过的图像
        :param img0: opencv读取的原始图像
        :param t0: 开始时间
        :param t1: 图片预处理后时间
        :param t2: 推理时间
        :param t3: nms时间
        :return: 处理过的检测结果(YoloResult)
        """
        results = []
        for i, det in enumerate(pred):  # detections per image
            s = ''

            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf)
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=1)
                    name = self.names[int(cls)]
                    box = [xyxy[:][0].item(), xyxy[:][1].item(), xyxy[:][2].item(), xyxy[:][3].item()]
                    center = self.xyxy2cnt(box)
                    result = YoloResult(name, box, center[0], center[1], float(conf))
                    results.append(result)

            # Print time (inference + NMS)
            # print(f'{s}fps:{(1 / (t3 - t0)):.2f}. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        return results

    def pred(self, model, img0: cv2_image) -> List[YoloResult]:
        """模型推理

        :param model: 调用load_model加载的模型
        :param img0: opencv读取的原始图像
        :return: 处理过的检测结果
        """
        t0 = time_synchronized()
        img = self.process_img(img0)
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=False)
        t3 = time_synchronized()
        results = self.process_result(pred, img, img0, t0, t1, t2, t3)
        return results

    def xyxy2cnt(self, xyxy: List[int]) -> List[int]:
        """
        xyxy坐标转中心点坐标

        :param xyxy: [x,y,x,y]
        :return: [x,y]
        """
        center = [int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2)]
        return center

    def show(self, img: cv2_image, results: List[YoloResult], resolution: List[int] = (640, 480),
             range: float = 1.0) -> NoReturn:
        """
        opencv显示图片

        :param img: opencv读取的原始图片
        :param results: List of YoloResult(每个YoloResult代表一个物品)
        :param resolution: 图片分辨率[宽,高]
        :param range: 画面中心多大范围内的检测结果被采用
        :return: None
        """
        cv2.namedWindow('yolo', cv2.WINDOW_NORMAL)
        width = resolution[0]
        height = resolution[1]
        cv2.resizeWindow('yolo', width, height)
        cv2.line(img, (int(width * 0.5 * (1 - range)), 0), (int(width * 0.5 * (1 - range)), height),
                 (0, 255, 0), 1, 4)
        cv2.line(img, (int(width * 0.5 * (1 + range)), 0), (int(width * 0.5 * (1 + range)), height),
                 (0, 255, 0), 1, 4)
        for item in results:
            cv2.circle(img, (int(item.x), int(item.y)), 1, (0, 0, 255), 8)
        cv2.imshow("yolo", img)

    def in_range(self, xs: List[int], width: int = 640, range: float = 1.0) -> bool:
        """
        判断物品中心点是否在设定的画面范围内

        :param xs: 一张图像上检测到的物品中心x坐标
        :param width: 图片宽度
        :param range: 画面中心多大范围内的检测结果被采用
        :return:
        """
        left = width * 0.5 * (1 - range)
        right = width * 0.5 * (1 + range)
        return any([left <= x <= right for x in xs])

    def judge(self, mode: str, results: List[YoloResult], width: int = 640, range: float = 1.0,
              find: str = None) -> bool:
        """
        判断是否退出检测

        :param mode: “realtime”(实时检测,按q退出) / “find”(检测到指定物品在画面范围内退出) / other(检测到任意物体在范围内退出)
        :param results: List of YoloResult(每个YoloResult代表一个物品)
        :param width: 图片宽度
        :param range: 画面中心多大范围内的检测结果被采用
        :param find: 需要寻找的物品名称
        :return: True/False
        """
        items = {item.name: item.x for item in results}
        if find is not None:
            mode = "find"
        if mode == 'realtime':
            flag = cv2.waitKey(1) & 0xFF == ord('q')
        elif mode == 'find':
            flag = cv2.waitKey(1) and find in items and self.in_range([items[find]], width, range)
        else:
            flag = cv2.waitKey(1) and items != [] and self.in_range([item.x for item in results], width, range)
        return flag

    def run(self, device: str = "cam", mode: str = "realtime", range: float = 1.0, nosave: bool = False,
            find: str = None, classes: str = None, rotate=1.0) -> List[YoloResult]:
        """
        运行检测

        :param device: 使用的设备(azure kinect/电脑摄像头)
        :param mode: “realtime”(实时检测,按q退出) / “find”(检测到指定物品在画面范围内退出) / other(检测到任意物体在范围内退出)
        :param range: 画面中心多大范围内的检测结果被采用
        :param nosave: 是否保存图片到本地
        :param find: 需要寻找的物体名称
        :param classes: 哪些物体可以被检测到,字符串名称间用','分割(例:传"bottle,person"则只会检测到bottle和person)
        :return: List of YoloResult(每个YoloResult代表一个物品)
        """
        model = self.load_model()
        if classes is not None:
            self.classes = [self.names.index(name) for name in classes.split(',')]
        results = []
        if device == "k4a" or device == "kinect":
            print("using k4a")


            # Initialize the library, if the library is not found, add the library path as argument
            pykinect.initialize_libraries()

            # Modify camera configuration
            device_config = pykinect.default_configuration
            device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
            # print(device_config)

            # Start device
            device = pykinect.start_device(config=device_config)
            cv2.namedWindow('Color Image',cv2.WINDOW_NORMAL)
            while True:
                if rotate:
                    self.base.rotate(rotate)

                # Get capture
                capture = device.update()

                # Get the color image from the capture
                ret, color_image = capture.get_color_image()
                if not ret:
                    continue
                img0 = color_image
                # resolution = [1280, 720]
                resolution = [1980, 1080]
                results = self.pred(model, img0)
                for item in results:
                    print(item)
                print("!----------------------------------------------------!")
                self.show(img0, results, resolution, range)
                print('show')
                if self.judge(mode, results, resolution[0], range, find=find):
                    if not nosave:
                        cv2.imwrite(str(Path(self.save_dir, "result.jpg")), img0)
                    break
            
            cv2.destroyAllWindows()
            if rotate:
                self.base.stop()
                print('base_stop_k4a')

            device.stop_cameras()
            device.close()

            # 释放图像                          
            # _k4a.k4a_capture_release()

            # _k4a.k4a_device_stop_cameras()
            # _k4a.k4a_device_close()

  

        else:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.open(0)
            # rate = rospy.Rate(2000)
            while cap.isOpened():
                flag, img0 = cap.read()
                resolution = [img0.shape[1], img0.shape[0]]
                results = self.pred(model, img0)
                for item in results:
                    print(item)
                print("----------------------------------------------------")
                self.show(img0, results, resolution, range)
                print('show')
                if self.judge(mode, results, resolution[0], range, find=find):
                    print('judge')
                    print(nosave)
                    if not nosave:
                        cv2.imwrite(str(Path(self.save_dir, "result.jpg")), img0)
                    break
            
            if rotate:
                self.base.stop()
                print('base_stop_k4a')
            cap.release()
        return results

    def detect(self, device: str = "cam", mode: str = "other", range: float = 1.0, nosave: bool = False,
               find: str = None, classes: str = None, rotate=0.5) -> List[YoloResult]:
        """
         运行检测

        :param device: 使用的设备(azure kinect/电脑摄像头)
        :param mode: “realtime”(实时检测,按q退出) / “find”(检测到指定物品在画面范围内退出) / other(检测到任意物体在范围内退出)
        :param range: 画面中心多大范围内的检测结果被采用
        :param nosave: 是否保存图片到本地
        :param find: 需要寻找的物体名称
        :param classes: 哪些物体可以被检测到,字符串名称间用','分割(例:传"bottle,person"则只会检测到bottle和person)
        :return: List of YoloResult(每个YoloResult代表一个物品)
        """
        # classes = "bottle,mouse,cell phone,cup"
        with torch.no_grad():  # 推理时减少显存占用
            results = self.run(device, mode, range, nosave, find, classes,rotate)
        return results


def main():
    weights_path = Path("weights")
    weights = Path(weights_path, "yolov7.pt")
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=weights, help='model.pt path(s)')
    parser.add_argument('--mode', type=str, default='realtime', help='detect mode')
    parser.add_argument('--range', type=float, default=1.0, help='detect range')
    parser.add_argument('--find', type=str, default=None, help='find object')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', type=str, default=None, help='filter by class: --class 0, or --class 0 2 3')
    opt = parser.parse_args()
    detector = ObjectDetector(opt.weights, opt.imgsz, opt.conf_thres, opt.iou_thres)
    results = detector.detect(device=opt.device, mode=opt.mode, range=opt.range, nosave=opt.nosave, find=opt.find,
                              classes=opt.classes)
    print("Detection finished!")
    for item in results:
        print(item)
    return results


if __name__ == '__main__':
    try:
        rospy.init_node('name', anonymous=True)

        # main()

        face = FaceDetector()
        result1 = face.detect(attributes=['age', 'gender', 'glasses', 'beauty', 'mask'], device='k4a')
        print(result1)
        
        # body = BodyDetector()
        # result2 = body.detect(attributes=['age', 'gender', 'upper_wear', 'upper_wear_texture', 'upper_wear_fg', 'upper_color',
        #      'lower_wear', 'lower_color', 'face_mask', 'glasses', 'headwear'], device='k4a')
        # print(result2)
        # #
        # yolo = ObjectDetector()
        # result = yolo.detect(device='cam', mode='realtime', find=None, rotate=1.0, range=0.5)
        # for object in result:
        #     print("name:{},box:{},x:{},y:{}".format(object.name, object.box, object.x, object.y))

    except rospy.ROSInterruptException:
        pass
