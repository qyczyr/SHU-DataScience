# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:39:29 2015

@author: young
"""
import pandas as pd
from sklearn import neighbors
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def load_data(path):
    df = pd.read_csv(path)
    return df

def save_result(path):
    pass

def kd_tree(test_data):
    pass
    '''
    print "start*****trainning"
    clf = neighbors.KNeighborsClassifier(algorithm='kd_tree')
    clf.fit(train_x, train_y)
    print "start*****predict"
    test_label=clf.predict(test_data)
    result = pd.DataFrame(test_label)
    print "save****result"
    result.to_csv("F:/datacastle/sklearn_knn_result.csv")
    '''
def random_forest_class(train_x,train_y,test_data):
    print "start*****trainning"
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(train_x, train_y)
    print "start*****predict"
    test_label = clf.predict(test_data)
    result = pd.DataFrame(test_label)
    print "save****result"
    result.to_csv("F:/datacastle/sklearn_random_forest_result.csv")

def statistic_category(train_data, features_type_data):
    category_index = [sample[0] for sample in features_type_data if sample[-1] == 'category']
    category_value = []   
    for index in category_index:
        category_value.append({'{}'.format(index):set(train_x[index])})
    return category_value
    
if __name__ == '__main__':
    train_x = load_data("./data/train_x.csv")
    features_type_data = np.array(load_data("./data/features_type.csv"))
    category = statistic_category(train_x, features_type_data)

    
    

    
    
    
            
    #a = [[].append( for each_data in list(features_type_data['type']) if each_data == "category")]
    

    '''
    #train_x = load_data("./data/train_x.csv")
    
    
    #train_x = np.array(train_x.iloc[:,1:-1])
    #train_y = load_data("./data/train_y.csv")
    #train_y = np.array(train_y['y'])
    #test_data = load_data("./data/test_x.csv")
    #test_data = np.array(test_data.iloc[:,1:-1])
    
    
    #saveResult(testLabel,'./sklearn_knn_Result.csv')
    '''    
    
    
