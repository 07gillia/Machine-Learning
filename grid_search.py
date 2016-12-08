from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

print(__doc__)

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

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