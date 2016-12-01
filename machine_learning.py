#####################################################################

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

#####################################################################
# shuffle, used to read in the data and then shuffle it to get 
# different sets of test data vs training data

def shuffle():

	########### Define classes

	classes = {} # define mapping of classes
	inv_classes = {v: k for k, v in classes.items()}

	########### Load Data Set

	path_to_data = "Dataset" 
	# the file that the data is in

	attribute_list = []
	# the temp variable to store the attributes 
	label_list = []
	# the temp variable to store the labels

	reader=csv.reader(open(os.path.join(path_to_data, "x.txt"),"rt", encoding='ascii'),delimiter=' ')
	# read in the attributes file
	for row in reader:
		# attributes in columns 0-561
		attribute_list.append(list(row[i] for i in (range(0,561))))
		# store in the temp list

	reader=csv.reader(open(os.path.join(path_to_data, "y.txt"),"rt", encoding='ascii'),delimiter=' ')
	# read in the labels file
	for row in reader:
		# attributes in column 1
		label_list.append(row[0])
		# store in the temp list

	attributes=np.array(attribute_list).astype(np.float32)
	labels=np.array(label_list).astype(np.float32)
	# put both in more perminant storage

	###########  test output for sanity
	"""
	print(attributes)
	print(len(attributes))
	print(labels)
	print(len(labels))
	"""
	#####################################################################

	########### Zip the two lists together so as to shuffle the data whist maintaining the realationship between the lists

	temp = list(zip(labels, attributes))
	# zip together

	random.shuffle(temp)
	# shuffle the data

	labels, attributes = zip(*temp)
	# unzip and restore

	#####################################################################

	########### Split Dataset into test and training

	N = int(len(labels) * 0.3)
	# spilt the data into 30% 

	test_labels = np.array(labels[:N]).astype(np.float32)
	test_attributes = np.array(attributes[:N]).astype(np.float32)
	# take the first 30% and use it for testing

	training_labels = np.array(labels[N:]).astype(np.float32)
	training_attributes = np.array(attributes[N:]).astype(np.float32)
	# take the last 70% and use it for training

	return test_labels, test_attributes, training_labels, training_attributes

#####################################################################

def kNN(training_labels, training_attributes, test_labels, test_attributes, K):
	############ Perform Training -- k-NN

	# define kNN object

	knn = cv2.ml.KNearest_create();

	# set to use BRUTE_FORCE neighbour search as KNEAREST_KDTREE seems to  break
	# on this data set (may not for others - http://code.opencv.org/issues/2661)

	knn.setAlgorithmType(cv2.ml.KNEAREST_BRUTE_FORCE);

	# set default 3, can be changed at query time in predict() call

	knn.setDefaultK(K);

	# set up classification, turning off regression

	knn.setIsClassifier(True);

	# perform training of k-NN

	knn.train(training_attributes, cv2.ml.ROW_SAMPLE, training_labels);

	############ Perform Testing -- k-NN

	correct = 0
	incorrect = 0

	for i in range(len(test_attributes)):
		# iterate though every index of the test data

		sample = np.vstack((test_attributes[i,:], np.zeros(len(test_attributes[i,:])).astype(np.float32)))
		# formatting before running

		_, results, neigh_respones, distances = knn.findNearest(sample, k = 3);
		# run the test on the current thing

		if(results[0] == test_labels[i]):
			correct += 1
		else:
			incorrect += 1

	print("Correct: " + str(correct/len(test_attributes) * 100))
	print("Incorrext: " + str(incorrect/len(test_attributes) * 100))
	return (correct/len(test_attributes) * 100)

#####################################################################

def SVM(training_labels, training_attributes, test_labels, test_attributes):

	use_svm_autotrain = False;

	############ Perform Training -- SVM

	# define SVM object

	svm = cv2.ml.SVM_create();

	# set kernel
	# choices : # SVM_LINEAR / SVM_RBF / SVM_POLY / SVM_SIGMOID / SVM_CHI2 / SVM_INTER

	svm.setKernel(cv2.ml.SVM_LINEAR);

	# set parameters (some specific to certain kernels)

	svm.setC(1.0); # penalty constant on margin optimization
	svm.setType(cv2.ml.SVM_C_SVC); # multiple class (2 or more) classification
	svm.setGamma(0.5); # used for SVM_RBF kernel only, otherwise has no effect
	svm.setDegree(3);  # used for SVM_POLY kernel only, otherwise has no effect

	# set the relative weights importance of each class for use with penalty term

	svm.setClassWeights(np.float32([1,1,1,1,1,1,1,1,1,1,1,1]));

	# define and train svm object

	if (use_svm_autotrain):

		# use automatic grid search across the parameter space of kernel specified above
		# (ignoring kernel parameters set previously)

		# if it is available : see https://github.com/opencv/opencv/issues/7224

		svm.trainAuto(cv2.ml.TrainData_create(training_attributes, cv2.ml.ROW_SAMPLE, training_labels.astype(int)), kFold=10);
	else:

		# use kernel specified above with kernel parameters set previously

		svm.train(training_attributes, cv2.ml.ROW_SAMPLE, training_labels.astype(int));

	############ Perform Testing -- SVM

	correct = 0 # handwritten digit correctly identified
	incorrect = 0   # handwritten digit wrongly identified

	# for each testing example

	for i in range(0, len(test_attributes[:,0])):

		# (to get around some kind of OpenCV python interface bug, vertically stack the
		#  example with a second row of zeros of the same size and type which is ignored).

		sample = np.vstack((test_attributes[i,:],
		np.zeros(len(test_attributes[i,:])).astype(np.float32)));

		# perform SVM prediction (i.e. classification)

		_, result = svm.predict(sample, cv2.ml.ROW_SAMPLE);

		if(test_labels[i] == result[0]):
			correct += 1
		else:
			incorrect += 1

	print("Correct: " + str(correct/len(test_attributes) * 100))
	print("Incorrext: " + str(incorrect/len(test_attributes) * 100))
	return correct/len(test_attributes) * 100

#####################################################################

# run the machine learning code here

# kNN
"""
for x in range(3,16):
	print("")
	print("kNN Round " + str(x))
	total = 0
	for y in range(0,10):
		test_labels, test_attributes, training_labels, training_attributes = shuffle()
		total += kNN(training_labels, training_attributes, test_labels, test_attributes,x)
	print("the total percentage: " + str(total / 10))
"""
"""
k = 3 = 94.9695
k = 4 = 95.01
k = 5 = 95.207
k = 6 = 94.95
k = 7 = 94.99
k = 8 = 94.97
k = 9 = 95.03
k = 10 = 94.88
k = 11 = 94.92
k = 12 = 95.15
k = 13 = 94.83
k = 14 = 95.06
k = 15 = 95.10
"""

# kNN with weighting

# SVM
counter = 0
for x in range(0,10):
	print("")
	print("SVM round " + str(x + 1))
	test_labels, test_attributes, training_labels, training_attributes = shuffle()
	counter += SVM(training_labels, training_attributes, test_labels, test_attributes)
print("Total Score: " + str(counter/10))
"""
setKernel = SVM_LINEAR
setC = 1.0
setType = SVM_C_SVC
setGamma = 0.5
setDegree = 3
Unweighted
SCORE = 96.09
"""

#####################################################################