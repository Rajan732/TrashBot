#!/usr/bin/env python

import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys

bridge= CvBridge()

video_capture= cv2.VideoCapture(0)

def video_publish(video):
    rospy.init_node("video_capture")
    pub=rospy.Publisher("detection", Image,queue_size=5)
    while(video.isOpened() and not rospy.is_shutdown()):
            ret, frame=video.read()
            try:
                ret,frame = video.read()
                if ret==True:
                    img = bridge.cv2_to_imgmsg(frame,'bgr8')
                    pub.publish(img)
                    print('published size of image %s'%len(frame))
            except CvBridgeError as e:
                print(e)
                break
        

if __name__=="__main__":
    video_publish(video_capture)