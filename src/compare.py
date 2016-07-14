#!/usr/bin/env python2

import roslib
roslib.load_manifest('hdr_cal2')

import dynamic_reconfigure.client
import glob
import rospy
from cv_bridge import CvBridge, CvBridgeError

import sys
import time

import cv2
import numpy as np

from exposure import Exposure
from camera import Camera

PROD = 1

def main():
    
    rospy.init_node('hdr_calibrator', anonymous=True)
    
    cameras = []
    clients = []
    
    if PROD:
    # Production
        camera3 = Camera(topic='cameras/stereo/left/image_raw', stride=1)

        cameras = [camera3]

        client3 = dynamic_reconfigure.client.Client("stereo_left_cam", timeout=3)

        clients = [client3]

    else:
        
        camera1 = Camera(topic='camera1/camera/image_raw', stride=1)
        cameras = [camera1]

        client1 = dynamic_reconfigure.client.Client("camera1/camera", timeout=3)
        clients = [client1]

    exposureControl = Exposure(cameras, clients)
    exposureControl.adjust()

    rospy.signal_shutdown("Calibration interrupted.")


if __name__ == '__main__':
    main()
