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
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

#####################################################################
# read in the files as two arrays that will then need to be split into train and test

def read_in_files():

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
# reads in the files and splits them based on K folding
# different sets of test data vs training data

def k_folding(K, splits, kNN, SVM, kNN_weighted):

	########### Split Dataset into test and training

	# split the data into sub sets, choose one at random, return all others as training data and the chosen as test data

	kf = KFold(n_splits=splits)

	total_correct = 0

	print(K)

	for train, test in kf.split(attributes):
		training_attributes, test_attributes, training_labels, test_labels = attributes[train], attributes[test], labels[train], labels[test]

		total_correct += kNN(training_labels, training_attributes, test_labels, test_attributes, K)

	print("Average: " + str(total_correct/10))

#####################################################################
# reads in the files and splits them based on K folding
# different sets of test data vs training data

def k_folding_stratified(K, splits,attributes):

	########### Split Dataset into test and training

	# split the data into sub sets, choose one at random, return all others as training data and the chosen as test data

	kf = StratifiedKFold(n_splits=splits)

	total_correct = 0

	print(K)

	for train, test in kf.split(attributes, labels):
		training_attributes, test_attributes, training_labels, test_labels = attributes[train], attributes[test], labels[train], labels[test]

		correct, predicted_labels,actual_labels = kNN(training_labels, training_attributes, test_labels, test_attributes, K)

		total_correct += correct

	print("Average: " + str(total_correct/10))

	return actual_labels, predicted_labels

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

#####################################################################

def SVM(training_labels, training_attributes, test_labels, test_attributes, kernal_value, c_value, gamma_value, degree_value):

	use_svm_autotrain = False

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

	if (use_svm_autotrain):

		# use automatic grid search across the parameter space of kernel specified above
		# (ignoring kernel parameters set previously)

		# if it is available : see https://github.com/opencv/opencv/issues/7224

		svm.trainAuto(cv2.ml.TrainData_create(training_attributes, cv2.ml.ROW_SAMPLE, training_labels.astype(int)), kFold=10)
	else:

		# use kernel specified above with kernel parameters set previously

		svm.train(training_attributes, cv2.ml.ROW_SAMPLE, training_labels.astype(int))

	############ Perform Testing -- SVM

	correct = 0 # handwritten digit correctly identified
	incorrect = 0   # handwritten digit wrongly identified

	# for each testing example

	for i in range(0, len(test_attributes[:,0])):

		# (to get around some kind of OpenCV python interface bug, vertically stack the
		#  example with a second row of zeros of the same size and type which is ignored).

		sample = np.vstack((test_attributes[i,:],
		np.zeros(len(test_attributes[i,:])).astype(np.float32)))

		# perform SVM prediction (i.e. classification)

		_, result = svm.predict(sample, cv2.ml.ROW_SAMPLE)

		if(test_labels[i] == result[0]):
			correct += 1
		else:
			incorrect += 1

	print("Correct: " + str(correct/len(test_attributes) * 100))
	print("Incorrext: " + str(incorrect/len(test_attributes) * 100))
	return correct/len(test_attributes) * 100

#####################################################################

def kNN_weighted(training_labels, training_attributes, test_labels, test_attributes, K, inverse_square, similarity):
	
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

	for i in range(len(test_attributes)):
		# iterate though every index of the test data

		sample = np.vstack((test_attributes[i,:], np.zeros(len(test_attributes[i,:])).astype(np.float32)))
		# formatting before running

		_, results, neigh_respones, distances = knn.findNearest(sample, k = K)
		# run the test on the current thing

		################### The weighting bit

		# Inverse Square Distance

		if(inverse_square):

			prediction = 0

			weighted_labels = np.array([0,0,0,0,0,0,0,0,0,0,0,0]).astype(np.float32)

			for x in range(len(neigh_respones[0])):
				# iterate through each of the neghbors and their decisions and distances
				# print(neigh_respones[0][x])
				# print(distances[0][x])

				current_inverse_square = 1/(distances[0][x] * distances[0][x])

				weighted_labels[neigh_respones[0][x] - 1] += current_inverse_square

			prediction = np.argmax(weighted_labels, axis=0) + 1

		# Similarity

		if(similarity):

			prediction = 0

			weighted_labels = np.array([0,0,0,0,0,0,0,0,0,0,0,0]).astype(np.float32)

			for x in range(len(neigh_respones[0])):
				# iterate through each of the neghbors and their decisions and distances
				# print(neigh_respones[0][x])
				# print(distances[0][x])

				current_similarity = 1 - distances[0][x]

				weighted_labels[neigh_respones[0][x] - 1] += current_similarity

			prediction = np.argmax(weighted_labels, axis=0) + 1

		################### End of the weighting bit

		if(prediction == test_labels[i]):
			correct += 1
		else:
			incorrect += 1

	print("Correct: " + str(correct/len(test_attributes) * 100))
	print("Incorrext: " + str(incorrect/len(test_attributes) * 100))
	return (correct/len(test_attributes) * 100)

#####################################################################
# Plot the confusion matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#####################################################################

def plot_everything(actual_labels, predicted_labels):

	# Compute confusion matrix
	cnf_matrix = confusion_matrix(actual_labels, predicted_labels)
	np.set_printoptions(precision=2)

	class_names = ['1','2','3','4','5','6','7','8','9','10','11','12']

	# Plot non-normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names,
                      	title='Confusion matrix, without normalization')

	# Plot normalized confusion matrix
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      	title='Normalized confusion matrix')

	plt.show()

#####################################################################

def AccPreRecF1(actual_labels, predicted_labels):
	accuracy = accuracy_score(actual_labels, predicted_labels)
	precision = precision_score(actual_labels, predicted_labels, average='micro') 
	recall = recall_score(actual_labels, predicted_labels, average='micro') 
	f1 = f1_score(actual_labels, predicted_labels, average='micro') 
	return accuracy, precision, recall, f1

#####################################################################

def testing():

	attributes, labels = read_in_files()
	labels, attributes = shuffle(labels, attributes)


	# Split the dataset in two equal parts
	X_train, X_test, y_train, y_test = train_test_split(
	    attributes, labels, test_size=0.5, random_state=0)

	# Set the parameters by cross-validation
	tuned_parameters = [
	                    {'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000, 10000]},
	                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
	                    {'kernel': ['poly'], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4], 'degree': [0.5, 1.0, 1,5, 2.0, 2,5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5], 'C': [1, 10, 100, 1000, 10000]},
	                    {'kernel': ['sigmoid'], 'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
	                    ]

	# SVM_LINEAR / SVM_RBF / SVM_POLY / SVM_SIGMOID / SVM_CHI2 / SVM_INTER
	# svm.setC(c_value) # penalty constant on margin optimization
	# svm.setType(cv2.ml.SVM_C_SVC) # multiple class (2 or more) classification
	# svm.setGamma(gamma_value) # used for SVM_RBF kernel only, otherwise has no effect
	# svm.setDegree(degree_value)  # used for SVM_POLY kernel only, otherwise has no effect

	scores = ['accuracy']

	for score in scores:
	    print("# Tuning hyper-parameters for %s" % score)
	    print()

	    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
	                       scoring=score)
	    clf.fit(X_train, y_train)

	    print("Best parameters set found on development set:")
	    print()
	    print(clf.best_params_)
	    print()
	    print("Grid scores on development set:")
	    print()
	    means = clf.cv_results_['mean_test_score']
	    stds = clf.cv_results_['std_test_score']
	    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
	        print("%0.3f (+/-%0.03f) for %r"
	              % (mean, std * 2, params))
	    print()

	    print("Detailed classification report:")
	    print()
	    print("The model is trained on the full development set.")
	    print("The scores are computed on the full evaluation set.")
	    print()
	    y_true, y_pred = y_test, clf.predict(X_test)
	    print(classification_report(y_true, y_pred))
	    print()

	# Note the problem is too easy: the hyperparameter plateau is too flat and the
	# output model is the same for precision and recall with ties in quality.

#####################################################################

# run the machine learning code here
# Have implemented k-folds, and stratified k-fold

# kNN
"""
attributes, labels = read_in_files()
for x in range(3,16):
	print("")
	print("kNN Round " + str(x))
	total = 0
	for y in range(0,10):
		labels, attributes = shuffle(labels, attributes)
		test_labels, test_attributes, training_labels, training_attributes = split_30_70(labels, attributes)
		to_add, _ , _ = kNN(training_labels, training_attributes, test_labels, test_attributes,x)
		total += to_add
	print("the total percentage: " + str(total / 10))
"""
""" MAY NEED TO RE-TEST!!!!!!!!!!!
k = 3 = 94.97
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
"""
for x in range(3,16):
	print("")
	print("Weighted kNN Round " + str(x))
	total = 0
	for y in range(0,10):
		test_labels, test_attributes, training_labels, training_attributes = shuffle()
		total += kNN_weighted(training_labels, training_attributes, test_labels, test_attributes,x)
	print("the total percentage: " + str(total / 10))
"""
"""
inverse
k = 3 = 95.34
k = 4 = 95.42
k = 5 = 95.77
k = 6 = 
k = 7 = 
k = 8 = 
k = 9 = 
k = 10 = 
k = 11 = 
k = 12 = 
k = 13 = 
k = 14 = 
k = 15 = 
similarity
k = 3 = 
k = 4 = 
k = 5 = 
k = 6 = 
k = 7 = 
k = 8 = 
k = 9 = 
k = 10 = 
k = 11 = 
k = 12 = 
k = 13 = 
k = 14 = 
k = 15 = 
"""

# SVM
"""
counter = 0
for x in range(0,10):
	print("")
	print("SVM round " + str(x + 1))
	test_labels, test_attributes, training_labels, training_attributes = shuffle()
	counter += SVM(training_labels, training_attributes, test_labels, test_attributes)
print("Total Score: " + str(counter/10))
"""
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

testing()

#####################################################################