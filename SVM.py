#####################################################################

# SVM
# Alex Gillies
# LLLL76
# Python version: 3

#####################################################################

import csv
import cv2
import os
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import itertools

import IO

#####################################################################

def SVM(training_labels, training_attributes, test_labels, test_attributes, kernal_value, c_value, gamma_value, degree_value):

	############ Perform Training -- SVM

	# define SVM object

	svm = cv2.ml.SVM_create()

	# set kernel
	# choices : # SVM_LINEAR / SVM_RBF / SVM_POLY / SVM_SIGMOID / SVM_CHI2 / SVM_INTER

	if(kernal_value == "SVM_LINEAR"):
		svm.setKernel(cv2.ml.SVM_LINEAR)
	if(kernal_value == "SVM_RBF"):
		svm.setKernel(cv2.ml.SVM_RBF)
	if(kernal_value == "SVM_POLY"):
		svm.setKernel(cv2.ml.SVM_POLY)
	if(kernal_value == "SVM_SIGMOID"):
		svm.setKernel(cv2.ml.SVM_SIGMOID)
	if(kernal_value == "SVM_CHI2"):
		svm.setKernel(cv2.ml.SVM_CHI2)
	if(kernal_value == "SVM_INTER"):
		svm.setKernel(cv2.ml.SVM_INTER)

	# set parameters (some specific to certain kernels)

	if(gamma_value == 0):
		gamma_value = 0.5
	if(degree_value == 0):
		degree_value = 3

	svm.setC(c_value) # penalty constant on margin optimization
	svm.setType(cv2.ml.SVM_C_SVC) # multiple class (2 or more) classification
	svm.setGamma(gamma_value) # used for SVM_RBF kernel only, otherwise has no effect
	svm.setDegree(degree_value)  # used for SVM_POLY kernel only, otherwise has no effect

	# set the relative weights importance of each class for use with penalty term

	svm.setClassWeights(np.float32([1,1,1,1,1,1,1,1,1,1,1,1]))

	# define and train svm object

	svm.train(training_attributes, cv2.ml.ROW_SAMPLE, training_labels.astype(int))

	############ Perform Testing -- SVM

	correct = 0 # handwritten digit correctly identified
	incorrect = 0   # handwritten digit wrongly identified

	actual_labels = []
	predicted_labels = []

	# for each testing example

	for i in range(0, len(test_attributes[:,0])):

		# (to get around some kind of OpenCV python interface bug, vertically stack the
		#  example with a second row of zeros of the same size and type which is ignored).

		sample = np.vstack((test_attributes[i,:],
		np.zeros(len(test_attributes[i,:])).astype(np.float32)))

		# perform SVM prediction (i.e. classification)

		_, result = svm.predict(sample, cv2.ml.ROW_SAMPLE)

		predicted_labels.append(int(result[0][0]))
		actual_labels.append(int(test_labels[i]))

		if(test_labels[i] == result[0]):
			correct += 1
		else:
			incorrect += 1

	print("Correct: " + str(correct/len(test_attributes) * 100))
	print("Incorrext: " + str(incorrect/len(test_attributes) * 100))
	return correct/len(test_attributes) * 100, predicted_labels, actual_labels
	# show or use results

#####################################################################

# Run Code Here

# this will run to show the best results for each of the functions

test_labels, test_attributes, training_labels, training_attributes = IO.read_in_everything()

SVM(training_labels, training_attributes, test_labels, test_attributes, "SVM_LINEAR", 1, 0, 0)

#####################################################################