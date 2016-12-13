#####################################################################

# All results and testing
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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

import IO
import kNN
import SVM

print("##########")

# this will run the three best of knn, weighted knn and svm each time this file is run
# this file will re-run and then output all data about each of these

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
    # decide to normalise the values or not

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # arrange the plt

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    # create and add the data to the plt

    # plt.tight_layout()
    # if used results in errors - looks pretty though
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # format the plt

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
	# show all the metrics used to judge the code
	accuracy = accuracy_score(actual_labels, predicted_labels)
	precision = precision_score(actual_labels, predicted_labels, average='micro') 
	recall = recall_score(actual_labels, predicted_labels, average='micro') 
	f1 = f1_score(actual_labels, predicted_labels, average='micro') 

	print("Accuracy: " + str(accuracy))
	print("Precision: " + str(precision))
	print("Recall: " + str(recall))
	print("F1: " + str(f1))
	return accuracy, precision, recall, f1

#####################################################################

def grid_search():

	attributes, labels = IO.read_in_files()
	labels, attributes = IO.shuffle(labels, attributes)

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

		if(kNN):
			total_correct += kNN(training_labels, training_attributes, test_labels, test_attributes, K) # change for the best 
		if(SVM):
			total_correct += SVM(training_labels, training_attributes, test_labels, test_attributes, K) # CHANGE FOR BEST VALUE
		if(kNN_weighted):
			total_correct += kNN.kNN_weighted(training_labels, training_attributes, test_labels, test_attributes, 3, True, False) # change for the best

	print("Average: " + str(total_correct/10))

#####################################################################
# reads in the files and splits them based on K folding
# different sets of test data vs training data

def k_folding_stratified(K, splits, attributes, kNN, SVM, kNN_weighted):

	########### Split Dataset into test and training

	# split the data into sub sets, choose one at random, return all others as training data and the chosen as test data

	kf = StratifiedKFold(n_splits=splits)

	total_correct = 0

	print(K)

	for train, test in kf.split(attributes, labels):
		training_attributes, test_attributes, training_labels, test_labels = attributes[train], attributes[test], labels[train], labels[test]

		if(kNN):
			correct, predicted_labels,actual_labels = kNN(training_labels, training_attributes, test_labels, test_attributes, K) # CHANGE FOR BEST VALUE
		if(SVM):
			correct = SVM(training_labels, training_attributes, test_labels, test_attributes, K) # CHANGE FOR BEST VALUE
		if(kNN_weighted):
			correct = kNN_weighted(training_labels, training_attributes, test_labels, test_attributes, 3, True, False) # change for best values

		total_correct += correct

	print("Average: " + str(total_correct/10))

	return actual_labels, predicted_labels

#####################################################################
##### grid search

def all_gird_search():

	grid_search()

	print("-----")
	print("kNN")

	for x in range(1,100):
		# iterate through a lot of values of K
		print("-----")
		print("value of k: " + str(x))
		total = 0
		for y in range(0,10):
			# do each one 10 times
			test_labels, test_attributes, training_labels, training_attributes = IO.read_in_everything()
			# with different test values
			to_add, _, _ = kNN.kNN(training_labels, training_attributes, test_labels, test_attributes,x)
			# accumulate the percentage of each
			total += to_add
		print("the total percentage: " + str(total / 10))
		# get the average

	print("-----")
	print("kNN Weighted Inverse")

	for x in range(1,100):
		# iterate through a lot of values of K
		print("-----")
		print("value of k: " + str(x))
		total = 0
		for y in range(0,10):
			# do each one 10 times
			test_labels, test_attributes, training_labels, training_attributes = IO.read_in_everything()
			# with different test values
			to_add, _, _ = kNN.kNN_weighted(training_labels, training_attributes, test_labels, test_attributes, x, True, False)
			# accumulate the percentage of each
			total += to_add
		print("the total percentage: " + str(total / 10))
		# get the average

	print("-----")
	print("kNN Weighted Similarity")

	for x in range(1,100):
		# iterate through a lot of values of K
		print("-----")
		print("value of k: " + str(x))
		total = 0
		for y in range(0,10):
			# do each one 10 times
			test_labels, test_attributes, training_labels, training_attributes = IO.read_in_everything()
			# with different test values
			to_add, _, _ = kNN.kNN_weighted(training_labels, training_attributes, test_labels, test_attributes, x, False, True)
			# accumulate the percentage of each
			total += to_add
		print("the total percentage: " + str(total / 10))
		# get the average

#####################################################################

# Run Code Here

# runs all the best values and shows their stuff, confusion matrix and accuracy and stuff
'''
test_labels, test_attributes, training_labels, training_attributes = IO.read_in_everything()

##### kNN Results

_, predicted_labels, actual_labels = kNN.kNN(training_labels, training_attributes, test_labels, test_attributes, 3)

AccPreRecF1(actual_labels, predicted_labels)

plot_everything(actual_labels, predicted_labels)

##### weighted kNN Results

_, predicted_labels, actual_labels = kNN.kNN_weighted(training_labels, training_attributes, test_labels, test_attributes, 3, True, False)

AccPreRecF1(actual_labels, predicted_labels)

plot_everything(actual_labels, predicted_labels)

##### SVM Results

_, predicted_labels, actual_labels = SVM.SVM(training_labels, training_attributes, test_labels, test_attributes, "SVM_LINEAR", 1, 0, 0)

AccPreRecF1(actual_labels, predicted_labels)

plot_everything(actual_labels, predicted_labels)
'''
##### other searching

print("-----")
print("kNN Weighted Similarity")

for x in range(1,100):
	# iterate through a lot of values of K
	print("-----")
	print("value of k: " + str(x))
	total = 0
	for y in range(0,10):
		# do each one 10 times
		test_labels, test_attributes, training_labels, training_attributes = IO.read_in_everything()
		# with different test values
		to_add, _, _ = kNN.kNN_weighted(training_labels, training_attributes, test_labels, test_attributes, x, False, True)
		# accumulate the percentage of each
		total += to_add
	print("the total percentage: " + str(total / 10))
	# get the average

print("-----")
print("kNN 80:20")

for x in range(1,100):
	# iterate through a lot of values of K
	print("-----")
	print("value of k: " + str(x))
	total = 0
	for y in range(0,10):
		# do each one 10 times
		test_labels, test_attributes, training_labels, training_attributes = IO.read_in_everything_20()
		# with different test values
		to_add, _, _ = kNN.kNN_weighted(training_labels, training_attributes, test_labels, test_attributes, x, False, True)
		# accumulate the percentage of each
		total += to_add
	print("the total percentage: " + str(total / 10))
	# get the average

print("-----")
print("kNN 90:10")

for x in range(1,100):
	# iterate through a lot of values of K
	print("-----")
	print("value of k: " + str(x))
	total = 0
	for y in range(0,10):
		# do each one 10 times
		test_labels, test_attributes, training_labels, training_attributes = IO.read_in_everything_10()
		# with different test values
		to_add, _, _ = kNN.kNN_weighted(training_labels, training_attributes, test_labels, test_attributes, x, False, True)
		# accumulate the percentage of each
		total += to_add
	print("the total percentage: " + str(total / 10))
	# get the average

#####################################################################