# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 14:34:40 2018

@author: Allouda
"""

#import librairies 
import pandas as pd
import numpy as np
from sklearn import neighbors, metrics
from gridSearch import GridSearch
from sklearn import preprocessing
import matplotlib.pyplot as plt


# read data
data = pd.read_csv('winequality-red.csv', sep=";")
X = data.as_matrix(data.columns[:-1])
y = data.as_matrix([data.columns[-1]])
y = y.flatten()
#y_class = np.where(y<6, 0, 1)
from sklearn import model_selection
X_train, X_test, y_train, y_test = \
	model_selection.train_test_split(X, y,
                                	test_size=0.3 # 30% of data
                                	)
    
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)


# hyperparametrs
param_grid = {'n_neighbors':[3, 5, 7, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23]}

print ("optimisation based on mean_squared_error")

clf = GridSearch(neighbors.KNeighborsRegressor(), # kNN classifier
           param_grid, # hyperparameter
           5, # folds 
           'neg_mean_squared_error' # score optimisation
           )

clf.fitting(X_train_std, y_train)

print ("Results of cross validation :")
for mean, neig in zip(clf.mean_squared_error_models, clf.param_grid['n_neighbors']):
    print ("MSE : %0.03f, neigbors : %3d " % (mean, neig))

print ("Best hyperparameter (s) :")
print (clf.best_param)

y_pred = clf.predict(X_test_std)
mse = metrics.mean_squared_error(y_test, y_pred)
print ("Error on the test set: {0:0.03f}".format(mse))
r2 = metrics.r2_score(y_test, y_pred)
print ("R2 on the test set : {0:0.03f}".format(r2))


sizes = {} # key: coordinates; value: number of points at these coordinates
for (yt, yp) in zip(list(y_test), list(y_pred)):
    if (yt, yp) in sizes:
        sizes[(yt, yp)] += 1
    else:
        sizes[(yt, yp)] = 1

keys = sizes.keys()
plt.scatter([k[0] for k in keys], # True value (abscisse)
[k[1] for k in keys], # pridcted value (ordonnee)
s=[sizes[k] for k in keys], # marker size
color='coral')
plt.title("Predicted vs. real values")
plt.xlabel("real values")
plt.ylabel("Predicted values")
plt.text(3.5, 6.5, "r2="+str("%0.2f" % r2))
plt.text(3.5, 6.2, "mse="+str("%0.2f" % mse))
plt.show()

print ("Optimisation basd on R2")
clf = GridSearch(neighbors.KNeighborsRegressor(), 
           param_grid, 
           5, 
           'r2'
           )

clf.fitting(X_train_std, y_train)

print ("Results of cross validation :")
for mean, neig in zip(clf.r2_models, clf.param_grid['n_neighbors']):
    print ("R2 : %0.03f, neigbors : %3d " % (mean, neig))

print ("Best hyperparameter (s) on the training set:")
print (clf.best_param)

y_pred = clf.predict(X_test_std)
mse = metrics.mean_squared_error(y_test, y_pred)
print ("Error on the training set : {0:0.03f}".format(mse))
r2 = metrics.r2_score(y_test, y_pred)
print ("R2 on test set : {0:0.03f}".format(r2))


sizes = {} # key: coordinates; value: number of points at these coordinates
for (yt, yp) in zip(list(y_test), list(y_pred)):
    if (yt, yp) in sizes:
        sizes[(yt, yp)] += 1
    else:
        sizes[(yt, yp)] = 1

keys = sizes.keys()
plt.scatter([k[0] for k in keys], 
[k[1] for k in keys], 
s=[sizes[k] for k in keys], 
color='coral')
plt.title("Predicted vs. real values")
plt.xlabel("real values")
plt.ylabel("Predicted values")
plt.text(3.5, 6.5, "r2="+str("%0.2f" % r2))
plt.text(3.5, 6.2, "mse="+str("%0.2f" % mse))
plt.show()

print ("Random Model  : ")
y_pred_random = np.random.randint(np.min(y), np.max(y), y_test.shape)
print ("On Test set : {0:0.03f}".format(metrics.mean_squared_error(y_test, y_pred_random)))
print ("R2 on test set : {0:0.03f}".format(metrics.r2_score(y_test, y_pred_random)))

print ("Regrosseur Dummy : ")

from sklearn import dummy
dum = dummy.DummyRegressor(strategy='mean')

# Training
dum.fit(X_train_std, y_train)

# predction
y_pred_dum = dum.predict(X_test_std)

# Evaluate
print ("on test set : {0:0.03f}".format(metrics.mean_squared_error(y_test, y_pred_dum)))
print ("R2 son test set : {0:0.03f}".format(metrics.r2_score(y_test, y_pred_dum)))

