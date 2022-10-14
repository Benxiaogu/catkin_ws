#! /usr/bin/python3
# coding: UTF-8


import sys
sys.path.insert(1, '../')
import numpy as np
from pyKinectAzure import pyKinectAzure, _k4a
from kinectBodyTracker import kinectBodyTracker, _k4abt
import cv2
import pykinect_azure as pykinect




def State_Judgement(skeleton2D):
    #预留接口，方便后续开发        
    skeleton_size = np.size(skeleton2D.joints2D)        
    skeleton_list = np.zeros((3,skeleton_size), dtype=np.float32)        
    for i in range(skeleton_size):            
        skeleton_list[0][i] = skeleton2D.joints2D[i].position.xy.x            
        skeleton_list[1][i] = skeleton2D.joints2D[i].position.xy.y            
        skeleton_list[2][i] = skeleton2D.joints2D[i].confidence_level    



if __name__ == "__main__":    
    modulePath = r'/usr/lib/x86_64-linux-gnu/libk4a.so' 
    bodyTrackingModulePath = r'/usr/lib/libk4abt.so'
    # 使用包含模块的路径初始化库    
    pyK4A = pyKinectAzure(modulePath)    
    # 打开设备    
    pyK4A.device_open()    
    print('device_open')
    # 修改相机参数    
    device_config = pyK4A.config    
    device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_OFF   
    # device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1080P 
    device_config.depth_mode = _k4a.K4A_DEPTH_MODE_NFOV_UNBINNED    
    # print('device_config:',device_config)    
    # 使用修改的配置启动摄像头    
    pyK4A.device_start_cameras(device_config)    
    print('device_start')
    # 初始化身体跟踪器    
    # pyK4A.bodyTracker_start(bodyTrackingModulePath)   
    pykinect.initialize_libraries(track_body=True) 
    print('body_start')
    k = 0    
    while True:        
        # 获取捕获       
        pyK4A.device_get_capture()        
        # 从捕获中获取深度图像        
        depth_image_handle = pyK4A.capture_get_depth_image()        
        # 检查图像是否正确读取        
        if depth_image_handle:    
            print(depth_image_handle)        
            # 是否存在人体            
            pyK4A.bodyTracker_update()            
            # 读取图像数据并将其转换为numpy数组            
            depth_image = pyK4A.image_convert_to_numpy(depth_image_handle)            
            depth_color_image = cv2.convertScaleAbs (depth_image, alpha=0.05)  
            #alpha通过与Azure k4aviewer结果的视觉比较进行拟合            
            depth_color_image = cv2.cvtColor(depth_color_image, cv2.COLOR_GRAY2RGB)             
            # 获取人体分割图像            
            body_image_color = pyK4A.bodyTracker_get_body_segmentation()            
            combined_image = cv2.addWeighted(depth_color_image, 0.8, body_image_color, 0.2, 0)            
            # 画人体骨架            
            for body in pyK4A.body_tracker.bodiesNow:                
                skeleton2D = pyK4A.bodyTracker_project_skeleton(body.skeleton)                
                State_Judgement(skeleton2D)                
                combined_image = pyK4A.body_tracker.draw2DSkeleton(skeleton2D, body.id, combined_image)            
            # 基于深度图像的覆盖体分割            
            cv2.imshow('Segmented Depth Image',combined_image)            
            k = cv2.waitKey(1)            
            
            # 释放图像            
            pyK4A.image_release(depth_image_handle)            
            pyK4A.image_release(pyK4A.body_tracker.segmented_body_img)        
        pyK4A.capture_release()        
        pyK4A.body_tracker.release_frame()

        if k==27:    
            # 按Esc键停止            
            break        
        elif k == ord('q'):            
            cv2.imwrite('outputImage.jpg',combined_image)    
    pyK4A.device_stop_cameras()    
    pyK4A.device_close()