import rospy
from std_msgs.msg import String  # String类型消息,从String.data中可获得信息
import cv2
import sys
sys.path.insert(1, '../')
import pykinect_azure as pykinect

class exampleBodyTrackingLiteModel:
    def __init__(self):
        rospy.init_node('exampleBody', anonymous=True)  # 初始化ros节点
        rospy.Subscriber('/exampleBodyTrackingLiteModel', String, self.exampleBodyTrackingLiteModel)  # 创建订阅者订阅recognizer发出的地点作为启动信号
        self.startkinect = rospy.Publisher('/exampleBodyTracking', String, queue_size=1)
        self.pykinect.initialize_libraries=pykinect.initialize_libraries(track_body=True)
        self.device_config = pykinect.default_configuration
        self.device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
        self.device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
        self.device = pykinect.start_device(config=self.device_config)
        self.bodyTracker = pykinect.start_body_tracker(model_type=pykinect.K4ABT_DEFAULT_MODEL)
        self.cv2.namedWindow=cv2.namedWindow('Depth image with skeleton',cv2.WINDOW_NORMAL)

    def exampleBodyTrackingLiteModel(self):
        while True:
            capture =self. device.update()
            body_frame = self.bodyTracker.update()
            ret, depth_color_image = capture.get_colored_depth_image()
            ret, body_image_color = body_frame.get_segmentation_image()
        
            if not ret:
                continue
        
            combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
            combined_image = body_frame.draw_bodies(combined_image)
            cv2.imshow('Depth image with skeleton',combined_image)
            
            if cv2.waitKey(1) == ord('q'):  
                break

if __name__ == '__main__':
    try:
        exampleBodyTrackingLiteModel()  # 实例化Controller,参数为初始化ros节点使用到的名字
        rospy.spin()  # 保持监听订阅者订阅的话题
    except rospy.ROSInterruptException:
        pass