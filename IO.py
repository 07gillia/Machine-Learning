#####################################################################

# File input and Randomising 
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

#####################################################################
# read in the files as two arrays that will then need to be split into train and test

def read_in_files():

	########### Define classes

	classes = {} # define mapping of classes
	inv_classes = {v: k for k, v in classes.items()}

	########### Load Data Set

	path_to_data = "Dataset" 
	# the directory that the data is in

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
	
	return attributes, labels

#####################################################################
# shuffle, used to read in the data and then shuffle it to get 
# different sets of test data vs training data

def shuffle(labels, attributes):

	########### Zip the two lists together so as to shuffle the data whist maintaining the realationship between the lists

	temp = list(zip(labels, attributes))
	# zip together

	random.shuffle(temp)
	# shuffle the data

	labels, attributes = zip(*temp)
	# unzip and restore

	return labels, attributes

#####################################################################

def split_30_70(labels, attributes):

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

def split_20_80(labels, attributes):

	########### Split Dataset into test and training

	N = int(len(labels) * 0.2)
	# spilt the data into 20% 

	test_labels = np.array(labels[:N]).astype(np.float32)
	test_attributes = np.array(attributes[:N]).astype(np.float32)
	# take the first 20% and use it for testing

	training_labels = np.array(labels[N:]).astype(np.float32)
	training_attributes = np.array(attributes[N:]).astype(np.float32)
	# take the last 80% and use it for training

	return test_labels, test_attributes, training_labels, training_attributes

#####################################################################

def split_10_90(labels, attributes):

	########### Split Dataset into test and training

	N = int(len(labels) * 0.1)
	# spilt the data into 10% 

	test_labels = np.array(labels[:N]).astype(np.float32)
	test_attributes = np.array(attributes[:N]).astype(np.float32)
	# take the first 10% and use it for testing

	training_labels = np.array(labels[N:]).astype(np.float32)
	training_attributes = np.array(attributes[N:]).astype(np.float32)
	# take the last 90% and use it for training

	return test_labels, test_attributes, training_labels, training_attributes

#####################################################################

def read_in_everything():
	# this will call other funcitons in the file and has been tested and will use the best split (found experimentally)
	attributes, labels = read_in_files()
	labels, attributes = shuffle(labels, attributes)
	test_labels, test_attributes, training_labels, training_attributes = split_30_70(labels,attributes)

	return test_labels, test_attributes, training_labels, training_attributes

#####################################################################