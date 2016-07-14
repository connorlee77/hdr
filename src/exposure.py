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
from camera import Camera

DEBUG = 1

### Global Camera Settings ###
PIXELCOUNT = 752 * 480
BITSPERPIXEL = 8
# Voltage Options
VOLTAGE_OPTIONS = np.array([200, 400, 600, 800, 1000, 1200, 1761, 1784, 1808, 1831, 1855, 1878, 1902, 1949, 1972, 1996, 2019, 2043, 2066, 2090, 2113, 2137, 2160, 2184, 2207, 2231, 2254, 2278, 2301, 2325, 2348, 2372, 2395, 2419, 2442, 2466, 2489, 2513, 2536, 2583, 2607, 2630, 2654, 2677, 2701, 2724, 2748, 2771, 2795, 2818, 2842, 2865, 2889, 2912, 2936, 2959, 2983, 3006, 3030, 3053, 3077,3100])
# Exposure Times
SHUTTER_TIMES = np.array([500000, 250000, 125000, 62500, 31250, 15625, 7812, 3906, 1953, 976, 488, 244, 122, 61, 30])
SHUTTER_INIT = 0.0005

class Exposure:

	# Initializes an Exposure object.
	def __init__(self, cameras, clients):
		# Creates a camera that reads in every "stride" frames.
		self.cameras = cameras
		self.clients = clients
		self.num_cameras = float(len(cameras))
	
		# Initialize camera to initial parameters
		for i, client in enumerate(self.clients):
			client.update_configuration({
				"gain_auto" : False, 
				"shutter_auto" : False,
				"hdr_mode" : 'User'
			})


	# Retrieves a frame from the camera. Return the image's histogram and median.
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

	### Entropy methods ###
	# Entropy = -\Sum p * log(p) where
	# p is probability of each element in histogram occuring. 
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

	def tuneShutterSpeed(self, shutter_speed, TARGET_INTENSITY=120.0, epsilon=1.0):
			a = 0
			gradient = sys.maxint
			while a < 100 or gradient < epsilon:
				histogram, median = self.getHistogram(camera)
				gradient = self.intensityGradient(histogram, TARGET_INTENSITY, median)				
				shutter_speed *=np.exp(gradient * 0.001)

				self.clients[camera_idx].update_configuration({
					'shutter': shutter_speed
				})

				currAvg = self.getIntensityAverage(histogram, totalPixelCount=PIXELCOUNT, bitsPerPixel=BITSPERPIXEL)
				rospy.loginfo("Avg: " + str(currAvg))

				a += 1

			return shutter_speed

	def adjust(self):
		
		shutter_speed = SHUTTER_INIT

		v0_t = 0.0
		t0_t = 0.0
		v1_t = 0.0
		t1_t = 0.0
		shutter_speed_t = 0.0

		# Repeat for multiple camera
		for camera_idx, camera in enumerate(self.cameras):

			self.clients[camera_idx].update_configuration({
				'hdr_user_exposure_0': SHUTTER_TIMES[0],
				'hdr_user_voltage_0': 2889,
				"hdr_mode":'User',
				'hdr_user_kneepoint_count' : 1
			})

			histogram, median = self.getHistogram(camera)

			# Tune shutter speed according to the target intensity. 
			# Initialize shutter_speed at SHUTTER_INIT.
			# epsilon is convergence criteria for gradient.
			shutter_speed = self.tuneShutterSpeed(shutter_speed=shutter_speed, TARGET_INTENSITY=120.0, epsilon=1.0)

			# loop parameters
			i = 0 # loop counter
			n = 8 # number of loop iterations

			# Tuning parameters
			# Voltages initialized around 1/3 and 2/3 marks
			# for fixed point iterations.
			maxEntropy = 0			# Max entropy for adjusting kneepoint 0
			maxEntropy1 = 0			# Max entropy for adjusting kneepoint 1
			argmaxExposure = 0		# Exposure that maximizes entropy for kneepoint 0
			argmaxExposure1 = 0		# Exposure that maximizes entropy for kneepoint 1
			argmaxVoltage = 2889	# Voltage that maximizes entropy for kneepoint 0
			argmaxVoltage1 = 2066	# Voltage that maximizes entropy for kneepoint 1
				
			
			while i < n:
				
				# Adjust kneepoint 0 in first half of iterations.
				# Adjust kneepoint 1 in second half of iterations.
				if i < n / 2:

					# Alternate between adjusting kneepoint time parameter 
					# and voltage. 
					if i % 1 == 0:

						for exposure in SHUTTER_TIMES:
							
							if DEBUG:
								rospy.loginfo("Exposure: " + str(exposure))
							
							self.clients[camera_idx].update_configuration({
								'hdr_user_exposure_0' : exposure
							})
							
							histogram, median = self.getHistogram(camera)
							entropy = self.getEntropy(histogram)
							
							if DEBUG:														
								rospy.loginfo("entropy: " + str(entropy))
							
							if entropy >= maxEntropy:
								maxEntropy = entropy
								argmaxExposure = exposure

					else:

						for voltage in VOLTAGE_OPTIONS[15:]:
							
							if DEBUG:
								rospy.loginfo("Voltage: " + str(voltage))

							self.clients[camera_idx].update_configuration({
								'hdr_user_voltage_0' : voltage
							})
							histogram, median = self.getHistogram(camera)
							entropy = self.getEntropy(histogram)
							
							if DEBUG:														
								rospy.loginfo("entropy: " + str(entropy))

							if entropy >= maxEntropy:
								maxEntropy = entropy
								argmaxVoltage = voltage
				else:

					# Set kneepoint count to 2.Ensure prior/current 
					# settings are updated.
					self.clients[camera_idx].update_configuration({
							'hdr_user_exposure_0':argmaxExposure,
							'hdr_user_voltage_0': argmaxVoltage,
							'hdr_user_exposure_1':argmaxExposure1,
							'hdr_user_voltage_1': argmaxVoltage1,
							"hdr_mode":'User',
							'hdr_user_kneepoint_count' : 2})

					if i % 1 == 0:

						for exposure in SHUTTER_TIMES:
							
							if DEBUG:
								rospy.loginfo("Exposure: " + str(exposure))
							
							self.clients[camera_idx].update_configuration({
								'hdr_user_exposure_1' : exposure
							})
							
							histogram, median = self.getHistogram(camera)
							entropy = self.getEntropy(histogram)

							if DEBUG:														
								rospy.loginfo("entropy: " + str(entropy))
							
							if entropy >= maxEntropy1:
								maxEntropy1 = entropy
								argmaxExposure1 = exposure

					else:

						for voltage in VOLTAGE_OPTIONS[15:]:
							
							if DEBUG:
								rospy.loginfo("Voltage: " + str(voltage))
							
							self.clients[camera_idx].update_configuration({
								'hdr_user_voltage_1' : voltage
							})
							histogram, median = self.getHistogram(camera)
							entropy = self.getEntropy(histogram)

							if DEBUG:														
								rospy.loginfo("entropy: " + str(entropy))
							
							if entropy >= maxEntropy1:
								maxEntropy1 = entropy
								argmaxVoltage1 = voltage

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
						'hdr_user_voltage_0' : argmaxVoltage
					})
				else:
					self.clients[camera_idx].update_configuration({
						'hdr_user_exposure_1' : argmaxExposure1,
						'hdr_user_voltage_1' : argmaxVoltage1
					})

				rospy.loginfo("Iteration: " + str(i))
				i += 1			

			# Average out found parameters. 
			v0_t += argmaxVoltage / self.num_cameras
			v1_t += argmaxVoltage1 / self.num_cameras
			t0_t += argmaxExposure / self.num_cameras
			t1_t += argmaxExposure1 / self.num_cameras
			shutter_speed_t += shutter_speed / self.num_cameras

		# Averaging may lead to imprecise options. Find closest parameters. 
		v0_t = VOLTAGE_OPTIONS[np.abs(VOLTAGE_OPTIONS - v0_t).argmin()]
		v1_t = VOLTAGE_OPTIONS[np.abs(VOLTAGE_OPTIONS - v1_t).argmin()]
		t0_t = SHUTTER_TIMES[np.abs(SHUTTER_TIMES - t0_t).argmin()]
		t1_t = SHUTTER_TIMES[np.abs(SHUTTER_TIMES - t1_t).argmin()]

		if DEBUG:
			rospy.loginfo("shutter speed_t: " + str(shutter_speed_t))
			rospy.loginfo("v0_t: " + str(v0_t))
			rospy.loginfo("v1_t: " + str(v1_t))
			rospy.loginfo("t1_t: " + str(t1_t))
			rospy.loginfo("t0_t: " + str(t0_t))

		# Set all camera parameters to global average.
		for client in self.clients:
			client.update_configuration({
				'hdr_user_exposure_0' : t0_t,
				'hdr_user_exposure_1' : t1_t,
				'hdr_user_voltage_0' : v0_t,
				'hdr_user_voltage_1' : v1_t,
				'shutter' : shutter_speed_t,
				'hdr_mode':'User'})







