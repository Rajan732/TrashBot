import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridgeError, CvBridge
import numpy as np
import sys

bridge=CvBridge()

def img_callback(img):
    print ('received video')
    global bridge
    try:
        cv_image = bridge.imgmsg_to_cv2(img, "bgr8")
    except CvBridgeError as e:
        print(e)
    return cv_image

def main(args):
    rospy.init_node('image_display_from_rob', anonymous=True)
    image_sub = rospy.Subscriber("tennis_ball_image",Image, img_callback)
    try:
       rospy.spin()
    except KeyboardInterrupt:
       print("Shutting down")
    cv2.destroyAllWindows()