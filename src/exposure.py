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
from camera import Camera

DEBUG = 1

### Global Camera Settings ###
PIXELCOUNT = 752 * 480
BITSPERPIXEL = 8
# Voltage Options
VOLTAGE_OPTIONS = np.array([200, 400, 600, 800, 1000, 1200, 1761, 1784, 1808, 1831, 1855, 1878, 1902, 1949, 1972, 1996, 2019, 2043, 2066, 2090, 2113, 2137, 2160, 2184, 2207, 2231, 2254, 2278, 2301, 2325, 2348, 2372, 2395, 2419, 2442, 2466, 2489, 2513, 2536, 2583, 2607, 2630, 2654, 2677, 2701, 2724, 2748, 2771, 2795, 2818, 2842, 2865, 2889, 2912, 2936, 2959, 2983, 3006, 3030, 3053, 3077,3100])
# Exposure Times
SHUTTER_TIMES = np.array([250000, 125000, 62500, 31250, 15625, 7812, 3906, 1953, 976, 488, 244, 122, 61, 30])
SHUTTERMAX = np.float32(SHUTTER_TIMES[0])
# Normalized exposure times for PID controller and other methods
SHUTTER_TIMES_RAW = np.divide(SHUTTER_TIMES, SHUTTERMAX) 
SHUTTER_INIT = 0.0005

class Exposure:

	# Initializes an Exposure object.
	def __init__(self, cameras, clients, maxSaturatedPixels=100, numKneePoints=2):
		# Creates a camera that reads in every "stride" frames.
		self.cameras = cameras
		self.clients = clients

		self.maxSaturatedPixels = maxSaturatedPixels
		self.numKneePoints = numKneePoints
	
		# Initialize camera to initial parameters
		for i, client in enumerate(self.clients):
			client.update_configuration({
				"gain_auto" : False, 
				"shutter_auto" : False,
				"hdr_mode" : 'User'
			})

		# Keeps track of image saturation levels across time axis.
		self.saturations = [0]
		# Keep track of PID errors for integral component. Initialized to arbitrary reference value.
		self.PIDerrors = SHUTTER_TIMES_RAW[7]
		# Keep track of previous error for the PID controller. Used for derivitive step.
		self.prevError = 0

	# Retrieves a frame from the camera. Return the image's histogram.
	def getHistogram(self, camera):

		frame = camera.getFrame()
		while frame == None:
			frame = camera.getFrame()
		
		histogram = cv2.calcHist([frame],[0],None,[256],[0,256]).reshape(1, 256)[0]

		return histogram, np.median(frame)


	### Exposure methods ###

	# Calculates the mean sample value of the histogram. 
	def getIntensityAverage(self, histogram, totalPixelCount=PIXELCOUNT, bitsPerPixel=BITSPERPIXEL):
		D = 2**bitsPerPixel

		currSum = 0.0
		histSum = 0.0
		for i in xrange(0, D):
			currSum += (i + 1) * histogram[i]
			histSum += histogram[i]

		return currSum / histSum 
		#return (currSum - totalPixelCount) / (totalPixelCount * (D - 1))


	### Saturation methods ###

	# Calculate saturation of the 255 intensity level of the histogram.
	# Saturation within [0, 1]
	def getSaturation(self, histogram, bitsPerPixel=BITSPERPIXEL, totalPixelCount=PIXELCOUNT):
		D = 2**bitsPerPixel

		return histogram[D - 1] / np.float32(totalPixelCount)

	# Adjust voltage values depending on saturation. Scale the adjustments 
	# in order to align with the voltage values of the camera.
	def adjustSaturation(self, histogram, v0raw, v1raw, n=100, scale=1):
		saturationLevel = self.getSaturation(histogram)

		arr = np.array(self.saturations)

		scale=200

		if saturationLevel > np.max(arr):
			v0raw += 1 * scale
			v1raw += 1 * scale
		elif saturationLevel < np.min(arr):
			v0raw -= 1 * scale
			v1raw -= 1 * scale

		if len(arr) > 100:
			self.saturations = [0]

		self.saturations.append(saturationLevel)

		v0 = VOLTAGE_OPTIONS[np.abs(VOLTAGE_OPTIONS - v0raw).argmin()]
		v1 = VOLTAGE_OPTIONS[np.abs(VOLTAGE_OPTIONS - v1raw).argmin()]

		return v0raw, v1raw, v0, v1

	### Entropy methods ###

	# Calculates entropy of image. Adjusts first kneepoint to decrease entropy. gamma and eta need to be 
	# tuned to align with exposure values of the camera. 
	def adjustEntropy(self, histogram, t0, maxt0, prevEntropy, epsilon=2.0, gamma=0.5, eta=1.0):

		# Normalize histogram
		normalizedHistogram = histogram / np.sum(histogram)

		entropy = 0
		for p in normalizedHistogram:
			if p != 0:
				entropy += p * np.log(p) 

		entropy *= -1
		gradient = entropy - prevEntropy

		if gradient > epsilon:
			t0 *= np.exp(1.0 * gradient)

		return t0, entropy, gamma

	def getEntropy(self, histogram):
		normalizedHistogram = histogram / np.sum(histogram)

		entropy = 0
		for p in normalizedHistogram:
			if p != 0:
				entropy += p * np.log(p) 

		entropy *= -1

		return entropy

	def intensityGradient(self, histogram, true, median):

		currAvg = self.getIntensityAverage(histogram, totalPixelCount=PIXELCOUNT, bitsPerPixel=BITSPERPIXEL)

		return true - currAvg

	def adjust(self):
		for client in self.clients:
			client.update_configuration({
				'hdr_user_exposure_0': SHUTTER_TIMES[0],
				'hdr_user_voltage_0': 2889,
				"hdr_mode":'User',
				'hdr_user_kneepoint_count' : 1})

		v0_t = 0
		t0_t = 0
		v1_t = 0
		t1_t = 0
		shutter_speed_t = 0

		for camera_idx, camera in enumerate(self.cameras):
			histogram, median = self.getHistogram(camera)

			# Initialization params
			# v0raw = 2500
			# v1raw = 2000
			# prevEntropy = 0
			# t0 = SHUTTER_TIMES[len(SHUTTER_TIMES) - 5]
			# gamma = t0

			# exposureReference = SHUTTER_TIMES[2]
			# exposure = SHUTTER_TIMES[3]
			# alpha = SHUTTER_TIMES[3]

			# t1 = SHUTTER_TIMES[len(SHUTTER_TIMES) - 5]
			shutter_speed = SHUTTER_INIT

			t0, t1, v0, v1 = None, None, None, None

			i = 0
			n = 8
			maxEntropy = 0
			maxEntropy1 = 0
			argmaxExposure = 0
			argmaxExposure1 = 0
			argmaxVoltage1 = 2066
			argmaxVoltage = 2889	
			a = 0
			while a < 100:
				histogram, median = self.getHistogram(camera)
				TARGET_INTENSITY = 120.0
				gradient = self.intensityGradient(histogram, TARGET_INTENSITY, median)				
				shutter_speed *=np.exp(gradient * 0.001)

				self.clients[camera_idx].update_configuration({
					'shutter': shutter_speed
				})

				currAvg = self.getIntensityAverage(histogram, totalPixelCount=PIXELCOUNT, bitsPerPixel=BITSPERPIXEL)
				rospy.loginfo("Avg: " + str(currAvg))

				a += 1

			while i < n:
				
				
				
				#Gradient method: 
				# TARGET_INTENSITY = 120.0
				# gradient = self.intensityGradient(histogram, TARGET_INTENSITY, median)				
				# shutter_speed *=np.exp(gradient * 0.001)

				# v0raw, v1raw, v0, v1 = self.adjustSaturation(histogram, v0raw, v1raw, n=100, scale=200)

				# if i % 2 == 0:
				# 	t0, prevEntropy, gamma = self.adjustEntropy(histogram, t0, t1 / 2, prevEntropy, epsilon=0.04, gamma=gamma, eta=2.2)
				# 	t0 = SHUTTER_TIMES[np.abs(SHUTTER_TIMES - t0).argmin()]
				# else:
				# 	t1, prevEntropy, gamma = self.adjustEntropy(histogram, t1, t1 / 2, prevEntropy, epsilon=0.04, gamma=gamma, eta=2.0)
				# 	t1 = SHUTTER_TIMES[np.abs(SHUTTER_TIMES - t1).argmin()]

				if i < n / 2:
					if i % 1 == 0:

						for exposure in SHUTTER_TIMES:
							rospy.loginfo(exposure)
							
							self.clients[camera_idx].update_configuration({
								'hdr_user_exposure_0' : exposure,
							})
							
							histogram, median = self.getHistogram(camera)
							entropy = self.getEntropy(histogram)
							
							rospy.loginfo("entropy: " + str(entropy))
							if entropy >= maxEntropy:
								maxEntropy = entropy
								argmaxExposure = exposure

					else:

						for voltage in VOLTAGE_OPTIONS[15:]:
							rospy.loginfo(voltage)
							self.clients[camera_idx].update_configuration({
								'hdr_user_voltage_0' : voltage,
							})
							histogram, median = self.getHistogram(camera)
							entropy = self.getEntropy(histogram)
							
							if entropy >= maxEntropy:
								maxEntropy = entropy
								argmaxVoltage = voltage
				else:

					self.clients[camera_idx].update_configuration({
							'hdr_user_exposure_0':argmaxExposure,
							'hdr_user_voltage_0': argmaxVoltage,
							'hdr_user_exposure_1':argmaxExposure1,
							'hdr_user_voltage_1': argmaxVoltage1,
							"hdr_mode":'User',
							'hdr_user_kneepoint_count' : 2})

					if i % 1 == 0:

						for exposure in SHUTTER_TIMES:
							rospy.loginfo(exposure)
							
							self.clients[camera_idx].update_configuration({
								'hdr_user_exposure_1' : exposure,
							})
							
							histogram, median = self.getHistogram(camera)
							entropy = self.getEntropy(histogram)
							
							if entropy >= maxEntropy1:
								maxEntropy1 = entropy
								argmaxExposure1 = exposure

					else:

						for voltage in VOLTAGE_OPTIONS[15:]:
							rospy.loginfo(voltage)
							self.clients[camera_idx].update_configuration({
								'hdr_user_voltage_1' : voltage,
							})
							histogram, median = self.getHistogram(camera)
							entropy = self.getEntropy(histogram)
							
							if entropy >= maxEntropy1:
								maxEntropy1 = entropy
								argmaxVoltage1 = voltage


				### Tests here ###

				if DEBUG:			
					currAvg = self.getIntensityAverage(histogram, totalPixelCount=PIXELCOUNT, bitsPerPixel=BITSPERPIXEL)
					rospy.loginfo("Avg: " + str(currAvg))
					rospy.loginfo("median: " + str(median))
					rospy.loginfo("maxEntropy: " + str(maxEntropy))
					rospy.loginfo("argmaxVoltage: " + str(argmaxVoltage))
					rospy.loginfo("argmaxExposure: " + str(argmaxExposure))
					rospy.loginfo("argmaxVoltage1: " + str(argmaxVoltage1))
					rospy.loginfo("argmaxExposure1: " + str(argmaxExposure1))

				if i < n / 2:
					self.clients[camera_idx].update_configuration({
						'hdr_user_exposure_0' : argmaxExposure,
						'hdr_user_voltage_0' : argmaxVoltage,
						'hdr_mode':'User'})
				else:
					self.clients[camera_idx].update_configuration({
						'hdr_user_exposure_0' : argmaxExposure,
						'hdr_user_exposure_1' : argmaxExposure1,
						'hdr_user_voltage_0' : argmaxVoltage,
						'hdr_user_voltage_1' : argmaxVoltage1,
						'hdr_mode':'User'})

				rospy.loginfo(i)
				i += 1			

		# 	v0_t += v0
		# 	v1_t += v1 
		# 	t0_t += t0
		# 	t1_t += t1
		# 	#shutter_speed_t += shutter_speed

		# camera_nums = float(len(self.cameras))
		# v0_t /= camera_nums
		# v1_t /= camera_nums 
		# t0_t /= camera_nums
		# t1_t /= camera_nums
		# #shutter_speed_t /= camera_nums

		# v0_t = VOLTAGE_OPTIONS[np.abs(VOLTAGE_OPTIONS - v0_t).argmin()]
		# v1_t = VOLTAGE_OPTIONS[np.abs(VOLTAGE_OPTIONS - v1_t).argmin()]
		# t0_t = SHUTTER_TIMES[np.abs(SHUTTER_TIMES - t0_t).argmin()]
		# t1_t = SHUTTER_TIMES[np.abs(SHUTTER_TIMES - t1_t).argmin()]

		# if DEBUG:
		# 	#rospy.loginfo("shutter speed: " + str(shutter_speed))
		# 	rospy.loginfo("v0_t: " + str(v0_t))
		# 	rospy.loginfo("v1_t: " + str(v1_t))
		# 	rospy.loginfo("t1_t: " + str(t1_t))
		# 	rospy.loginfo("t0_t: " + str(t0_t))

		# for client in self.clients:
		# 	client.update_configuration({
		# 		'hdr_user_exposure_0' : t0_t,
		# 		'hdr_user_exposure_1' : t1_t,
		# 		#'hdr_user_voltage_0' : v0_t,
		# 		#'hdr_user_voltage_1' : v1_t,
		# 		#'shutter' : shutter_speed,
		# 		'hdr_mode':'User'})







