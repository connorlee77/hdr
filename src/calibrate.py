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
from matplotlib import pyplot as plt

from exposure import Exposure
from camera import Camera

PROD = 0

def main():
    
    rospy.init_node('hdr_calibrator', anonymous=True)
    
    cameras = []
    clients = []
    
    if PROD:
    # Production
        camera1 = Camera(topic='cameras/mono/left/image_raw', stride=1)
        camera2 = Camera(topic='cameras/mono/right/image_raw', stride=1)
        camera3 = Camera(topic='cameras/stereo/left/image_raw', stride=1)
        camera4 = Camera(topic='cameras/stereo/right/image_raw', stride=1)

        cameras = [camera1, camera2, camera3, camera4]

        client1 = dynamic_reconfigure.client.Client("mono_left_cam", timeout=3)
        client2 = dynamic_reconfigure.client.Client("mono_right_cam", timeout=3)
        client3 = dynamic_reconfigure.client.Client("stereo_left_cam", timeout=3)
        client4 = dynamic_reconfigure.client.Client("stereo_right_cam", timeout=3)

        clients = [client1, client2, client3, client4]

    else:
        
        camera1 = Camera(topic='camera1/camera/image_raw', stride=10)
        cameras = [camera1]

        client1 = dynamic_reconfigure.client.Client("camera1/camera", timeout=3)
        clients = [client1]

    exposureControl = Exposure(cameras, clients)
    exposureControl.adjust()

    rospy.signal_shutdown("Calibration interrupted.")


if __name__ == '__main__':
    main()
