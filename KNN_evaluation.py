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
                                	test_size=0.3 # 30% des données dans le jeu de test
                                	)
    
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)


# Fixer les valeurs des hyperparamètres à tester
param_grid = {'n_neighbors':[3, 5, 7, 9, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23]}

print ("optimisation a base de mean_squared_error")

clf = GridSearch(neighbors.KNeighborsRegressor(), # un classifieur kNN
           param_grid, # hyperparamètres à tester
           5, # nombre de folds de validation croisée
           'neg_mean_squared_error' # score à optimiser
           )

clf.fitting(X_train_std, y_train)

print ("Résultats de la validation croisée :")
for mean, neig in zip(clf.mean_squared_error_models, clf.param_grid['n_neighbors']):
    print ("MSE : %0.03f, neigbors : %3d " % (mean, neig))

print ("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:")
print (clf.best_param)

y_pred = clf.predict(X_test_std)
mse = metrics.mean_squared_error(y_test, y_pred)
print ("Error sur le jeu de test : {0:0.03f}".format(mse))
r2 = metrics.r2_score(y_test, y_pred)
print ("R2 sur le jeu de test : {0:0.03f}".format(r2))


sizes = {} # clé : coordonnées ; valeur : nombre de points à ces coordonnées
for (yt, yp) in zip(list(y_test), list(y_pred)):
    if (yt, yp) in sizes:
        sizes[(yt, yp)] += 1
    else:
        sizes[(yt, yp)] = 1

keys = sizes.keys()
plt.scatter([k[0] for k in keys], # vraie valeur (abscisse)
[k[1] for k in keys], # valeur predite (ordonnee)
s=[sizes[k] for k in keys], # taille du marqueur
color='coral')
plt.title("Valeurs prédites vs. valeurs réelles")
plt.xlabel("valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.text(3.5, 6.5, "r2="+str("%0.2f" % r2))
plt.text(3.5, 6.2, "mse="+str("%0.2f" % mse))
plt.show()

print ("Optimisation a base de R2")
clf = GridSearch(neighbors.KNeighborsRegressor(), # un classifieur kNN
           param_grid, # hyperparamètres à tester
           5, # nombre de folds de validation croisée
           'r2' # score à optimiser
           )

clf.fitting(X_train_std, y_train)

print ("Résultats de la validation croisée :")
for mean, neig in zip(clf.r2_models, clf.param_grid['n_neighbors']):
    print ("R2 : %0.03f, neigbors : %3d " % (mean, neig))

print ("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:")
print (clf.best_param)

y_pred = clf.predict(X_test_std)
mse = metrics.mean_squared_error(y_test, y_pred)
print ("Error sur le jeu de test : {0:0.03f}".format(mse))
r2 = metrics.r2_score(y_test, y_pred)
print ("R2 sur le jeu de test : {0:0.03f}".format(r2))


sizes = {} # clé : coordonnées ; valeur : nombre de points à ces coordonnées
for (yt, yp) in zip(list(y_test), list(y_pred)):
    if (yt, yp) in sizes:
        sizes[(yt, yp)] += 1
    else:
        sizes[(yt, yp)] = 1

keys = sizes.keys()
plt.scatter([k[0] for k in keys], # vraie valeur (abscisse)
[k[1] for k in keys], # valeur predite (ordonnee)
s=[sizes[k] for k in keys], # taille du marqueur
color='coral')
plt.title("Valeurs prédites vs. valeurs réelles")
plt.xlabel("valeurs réelles")
plt.ylabel("Valeurs prédites")
plt.text(3.5, 6.5, "r2="+str("%0.2f" % r2))
plt.text(3.5, 6.2, "mse="+str("%0.2f" % mse))
plt.show()








print ("Model aléatoire : ")
y_pred_random = np.random.randint(np.min(y), np.max(y), y_test.shape)
print ("Sur le jeu de test : {0:0.03f}".format(metrics.mean_squared_error(y_test, y_pred_random)))
print ("R2 sur le jeu de test : {0:0.03f}".format(metrics.r2_score(y_test, y_pred_random)))

print ("Regrosseur Dummy : ")

from sklearn import dummy
dum = dummy.DummyRegressor(strategy='mean')

# Entraînement
dum.fit(X_train_std, y_train)

# Prédiction sur le jeu de test
y_pred_dum = dum.predict(X_test_std)

# Evaluate
print ("Sur le jeu de test : {0:0.03f}".format(metrics.mean_squared_error(y_test, y_pred_dum)))
print ("R2 sur le jeu de test : {0:0.03f}".format(metrics.r2_score(y_test, y_pred_dum)))

""" la validation a base de MSE et a base de R2 donnent les mêmes resultat.
Par contre, les deux modéles sont meuilleurs par apport au modéle aléatoire et modéle de Dummy.

"""