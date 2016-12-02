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
# K folds implementation

def do_k_fold(labels, attributes, K):
	# split the data into sub sets, choose one at random, return all others as training data and the chosen as test data

	together = list(zip(labels, attributes))
	# zip together

	y = int(len(together) / K)
	# find the length of each subset

	chunks = [together[x:x+100] for x in range(0, len(together), y)]
	# store the chunks

	results = []

	for x in range(0, len(chunks)):
		# iterate through all chunks

		test_list = chunks[x]
		# store the test chunk

		training_list = []

		for y in range(0,len(chunks)):
			# iterate through all chunks

			if(x != y):
				# if the chunk isnt the test chunk

				training_list.append(chunks[y])
				# add the chunk to the training list

		results.append([test_list,training_list])

	return results
	# return a list of length k each containing a list of [test_list, training_list]

#####################################################################