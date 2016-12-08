#####################################################################

# kNN and its variants
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

def kNN(training_labels, training_attributes, test_labels, test_attributes, K):

	############ Perform Training -- k-NN

	# define kNN object

	knn = cv2.ml.KNearest_create()

	# set to use BRUTE_FORCE neighbour search as KNEAREST_KDTREE seems to  break
	# on this data set (may not for others - http://code.opencv.org/issues/2661)

	knn.setAlgorithmType(cv2.ml.KNEAREST_BRUTE_FORCE)

	# set default 3, can be changed at query time in predict() call

	knn.setDefaultK(K)

	# set up classification, turning off regression

	knn.setIsClassifier(True)

	# perform training of k-NN

	knn.train(training_attributes, cv2.ml.ROW_SAMPLE, training_labels)

	############ Perform Testing -- k-NN

	correct = 0
	incorrect = 0

	actual_labels = []
	predicted_labels = []

	for i in range(len(test_attributes)):
		# iterate though every index of the test data

		sample = np.vstack((test_attributes[i,:], np.zeros(len(test_attributes[i,:])).astype(np.float32)))
		# formatting before running

		_, results, neigh_respones, distances = knn.findNearest(sample, k = K)
		# run the test on the current thing

		predicted_labels.append(int(results[0][0]))
		actual_labels.append(int(test_labels[i]))

		if(results[0] == test_labels[i]):
			correct += 1
		else:
			incorrect += 1

	print("Correct: " + str(correct/len(test_attributes) * 100))
	print("Incorrext: " + str(incorrect/len(test_attributes) * 100))
	return (correct/len(test_attributes) * 100), predicted_labels, actual_labels
	# show or use results

#####################################################################

def kNN_weighted(training_labels, training_attributes, test_labels, test_attributes, K, inverse_square, similarity):
	
	############ Perform Training -- k-NN

	# define kNN object

	knn = cv2.ml.KNearest_create()

	# set to use BRUTE_FORCE neighbour search as KNEAREST_KDTREE seems to  break
	# on this data set (may not for others - http://code.opencv.org/issues/2661)

	knn.setAlgorithmType(cv2.ml.KNEAREST_BRUTE_FORCE)

	# use the parameter for K as default

	knn.setDefaultK(K)

	# set up classification, turning off regression

	knn.setIsClassifier(True)

	# perform training of k-NN

	knn.train(training_attributes, cv2.ml.ROW_SAMPLE, training_labels)

	############ Perform Testing -- k-NN

	correct = 0
	incorrect = 0

	actual_labels = []
	predicted_labels = []

	for i in range(len(test_attributes)):
		# iterate though every index of the test data

		sample = np.vstack((test_attributes[i,:], np.zeros(len(test_attributes[i,:])).astype(np.float32)))
		# formatting before running

		_, results, neigh_respones, distances = knn.findNearest(sample, k = K)
		# run the test on the current thing

		################### The weighting bit

		# Inverse Square Distance

		if(inverse_square):
			# use a parameter as a switch

			prediction = 0
			# set a default prediction

			weighted_labels = np.array([0,0,0,0,0,0,0,0,0,0,0,0]).astype(np.float32)
			# an array to store the probability of each of the classes

			for x in range(len(neigh_respones[0])):
				# iterate through each of the neghbors and their decisions and distances

				current_inverse_square = 1/(distances[0][x] * distances[0][x])
				# work out the inverse square distance for the current neighbour

				index = int(neigh_respones[0][x] - 1)
				# get the index of the class that the current nearest neighbour thinks is correct 

				weighted_labels[index] += current_inverse_square
				# incremenet the probability of that class with the weight that the current neighbour has

			prediction = np.argmax(weighted_labels, axis=0) + 1
			# get the overall prediction

		# Similarity

		if(similarity):

			prediction = 0
			# set a default prediction

			weighted_labels = np.array([0,0,0,0,0,0,0,0,0,0,0,0]).astype(np.float32)
			# an array to store the probability of each of the classes

			for x in range(len(neigh_respones[0])):
				# iterate through each of the neghbors and their decisions and distances

				current_similarity = 1 - distances[0][x]
				# work out the similarity of the nearest neighbour

				weighted_labels[neigh_respones[0][x] - 1] += current_similarity
				# increment the index of the class that the current neighbiour thinks is correct

			prediction = np.argmax(weighted_labels, axis=0) + 1
			# get the overall prediction

		################### End of the weighting bit

		predicted_labels.append(int(results[0][0]))
		actual_labels.append(int(test_labels[i]))

		if(prediction == test_labels[i]):
			# if the prediction is correct
			correct += 1
			# increment the number of correct 
		else:
			incorrect += 1
			# increment the number of incorrect

	print("Correct: " + str(correct/len(test_attributes) * 100))
	print("Incorrext: " + str(incorrect/len(test_attributes) * 100))
	return (correct/len(test_attributes) * 100), predicted_labels, actual_labels
	# show or use results

#####################################################################

# Run Code Here

# this will run to show the best results for each of the functions

test_labels, test_attributes, training_labels, training_attributes = IO.read_in_everything()

kNN(training_labels, training_attributes, test_labels, test_attributes, 3)

kNN_weighted(training_labels, training_attributes, test_labels, test_attributes, 3, True, False)

#####################################################################