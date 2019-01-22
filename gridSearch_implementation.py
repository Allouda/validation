# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:07:00 2018

@author: Allouda
"""
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score


class GridSearch:
    """ class modeling cross-validation """
    def __init__(self, model, param_grid, cv, scoring):
        self.model = model
        self.models = []
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_param = 0
        self.auc_models = []
        self.best_model = 0
        self.mean_squared_error_models = []
        self.r2_models = []
    
    def verifFolds(self, x_train, y_train):
        """ Function allows to divide data on folds """
        train_lenght = len(x_train[:,1])
        if (train_lenght // self.cv) < 2:
            print ("The number of folds is large compared to the data")
            return False
        else:
            return True
    def fitting (self, x_train, y_train):
        """ Function allow to train the model on the training data
        """
        #if self.verifFolds(x_train, y_train) and self.scoring == 'accuracy':
        if isinstance(self.model, type (KNeighborsClassifier(n_neighbors=4))):
            if self.verifFolds(x_train, y_train) and self.scoring == 'accuracy':
                for j in self.param_grid['n_neighbors']:
                    auc = 0
                    skf = StratifiedKFold(n_splits=self.cv)
                    KNC = KNeighborsClassifier(n_neighbors=j)
                    for train_index, test_index in skf.split(x_train, y_train):
                        new_x_train, new_x_test = x_train[train_index], x_train[test_index]
                        new_y_train, new_y_test = y_train[train_index], y_train[test_index]
                        KNC.fit(new_x_train, new_y_train)
                        auc = auc + KNC.score(new_x_test, new_y_test)
                    self.auc_models.append(auc/self.cv)
                    self.models.append(KNC)
                self.best_param = self.param_grid['n_neighbors'][self.auc_models.index(max(self.auc_models))]
                self.best_model = self.models[self.auc_models.index(max(self.auc_models))]
        elif isinstance(self.model, type (KNeighborsRegressor(n_neighbors=4))):
            if self.verifFolds(x_train, y_train) and self.scoring == 'neg_mean_squared_error':
                for j in self.param_grid['n_neighbors']:
                    error = 0
                    skf = StratifiedKFold(n_splits=self.cv)
                    KNR = KNeighborsRegressor(n_neighbors=j)
                    for train_index, test_index in skf.split(x_train, y_train):
                        new_x_train, new_x_test = x_train[train_index], x_train[test_index]
                        new_y_train, new_y_test = y_train[train_index], y_train[test_index]
                        KNR.fit(new_x_train, new_y_train)
                        y_pred = KNR.predict(new_x_test)
                        error = error + mean_squared_error (new_y_test, y_pred)
                    self.mean_squared_error_models.append(error/self.cv)
                    self.models.append(KNR)
                self.best_param = self.param_grid['n_neighbors'][self.mean_squared_error_models.index(min(self.mean_squared_error_models))]
                self.best_model = self.models[self.mean_squared_error_models.index(min(self.mean_squared_error_models))]
            
            if self.verifFolds(x_train, y_train) and self.scoring == 'r2':
                for j in self.param_grid['n_neighbors']:
                    r2_score_val = 0
                    skf = StratifiedKFold(n_splits=self.cv)
                    KNR = KNeighborsRegressor(n_neighbors=j)
                    for train_index, test_index in skf.split(x_train, y_train):
                        new_x_train, new_x_test = x_train[train_index], x_train[test_index]
                        new_y_train, new_y_test = y_train[train_index], y_train[test_index]
                        KNR.fit(new_x_train, new_y_train)
                        y_pred = KNR.predict(new_x_test)
                        r2_score_val = r2_score_val + r2_score (new_y_test, y_pred)
                    self.r2_models.append(r2_score_val/self.cv)
                    self.models.append(KNR)
                self.best_param = self.param_grid['n_neighbors'][self.r2_models.index(max(self.r2_models))]
                self.best_model = self.models[self.r2_models.index(max(self.r2_models))]
        else:
            print("model is not knn!!")
        
    def predict(self, x_test):
        return self.best_model.predict(x_test)