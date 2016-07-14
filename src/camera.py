#!/usr/bin/env python2

import roslib
roslib.load_manifest('hdr_cal2')

import dynamic_reconfigure.client
import glob
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String

import sys
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt


class Camera:

    def __init__(self, topic, stride=5):
        
        self.bridge = CvBridge()
        self.image_subcriber = rospy.Subscriber(topic, Image, self.callback)
        self.normalizedImage = None
        self.framesProcessed = stride
        self.stride = stride 

    def callback(self,data):

        if self.framesProcessed == 0:
        
            self.normalizedImage = self.bridge.imgmsg_to_cv2(data, "mono8")
            
            # reset frame stride
            self.framesProcessed = self.stride
        else:
            self.framesProcessed -= 1

    def getFrame(self):
        if self.framesProcessed == 0:
            return self.normalizedImage
        else:
            return None



