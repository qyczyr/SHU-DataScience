# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:39:29 2015

@author: young
"""
import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing

import xgboost as xgb

def load_data(path):
    df = pd.read_csv(path)
    return df


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
        category_value.append({'{}'.format(index):set(train_data[index])})
    return category_value

def svm_classify(X, Y, test_data):
    clf = svm.SVC(C=0.1,kernel='linear')
    clf.fit(X, Y)
    result = clf.predict(test_data)
    return result
    
def count_sample(src_y):
    label_y = np.array(src_y)[:,-1]
    count_one = 0;
    count_zero = 0;
    for i in label_y:
        if i == 1:
            count_one += 1
        elif i == 0:
            count_zero += 1
    print "count_one:"+str(count_one)
    print "count_zero:"+str(count_zero)

def decision_tree(X,Y,test_data,test_id):
    clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=None,
                                      min_samples_split=1, min_samples_leaf=1, 
                                      min_weight_fraction_leaf=0.0, max_features=None,
                                      random_state=None, max_leaf_nodes=None, class_weight=None)
    clf = clf.fit(X, Y)
    result = clf.predict(test_data)
    return result
    #save_result = pd.DataFrame(result)
    
    #save_result.to_csv("./result.csv")    
    
    #print "score:{}".format(clf.score(test_data,test_label))    
 
def decision_treeCV(X,Y,test_data):
    param_grid=[{'criterion':['gini','entropy'],
                 'min_samples_split':[1,2,3],'min_samples_leaf':[1,2,3]}]
    dt=GridSearchCV(tree.DecisionTreeClassifier(),param_grid,cv=5,n_jobs=2).fit(X,Y)
    
    print("best estimator:{}".format(dt.best_estimator_))
    print("best score:{:.2f}".format(dt.best_score_))
    print str(dt.score(test_data))
    
def random_forest(X,Y):
    pass

def extratreeclf(X,Y,test_data):   
    clf = ExtraTreesClassifier(criterion='gini',n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
    clf = clf.fit(X,Y)
    result = clf.predict(test_data)
    return result
    
    #scores = cross_val_score(clf, X, Y)
    #print scores.mean()  
    '''    
    param_grid = [{'n_estimators':[5,10,20,25,30],'criterion':['gini','entropy'],
                   'max_depth':[None,5,10,15,20,25,30],'min_samples_split':[1,2,3],
                   'min_samples_leaf':[1,2,3,4,5,6,7]}]

    grid = GridSearchCV(ExtraTreesClassifier(), param_grid)
    grid.fit(X, Y)
    print("grid:{}".format(grid))
    print("best estimator:{}".format(grid.best_estimator_.alpha))
    print("best score:{}".format(grid.best_score_))  
    '''    
    '''    
    scores = cross_val_score(clf, X, Y)
    print scores.mean() 
    '''
def extratreeclfTest(X,Y):
    clf = ExtraTreesClassifier(criterion='gini',n_estimators=50, max_depth=None, min_samples_split=1, random_state=0)
    scores = cross_val_score(clf, X, Y)
    print scores.mean() 
    

def xgb_clf(X, Y, test_data,test_label):
    dtrain = xgb.DMatrix(X,label=Y)
    dtest = xgb.DMatrix(test_data,label=test_label)
    param={
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'early_stopping_rounds':100,
    'scale_pos_weight': 1400.0/13458.0,
        'eval_metric': 'auc',
    'gamma':0.1,
    'max_depth':8,
    'lambda':550,
        'subsample':0.7,
        'colsample_bytree':0.4,
        'min_child_weight':3,
        'eta': 0.02,
    'seed':'random_seed',
    'nthread':7
    }


    watchlist = [(dtest,'eval'),(dtrain,'train')]
    num_round = 1000
    bst = xgb.train(param, dtrain, num_round,watchlist)
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    err = sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) / float(len(preds))
    print("err:{}".format(err))
    bst.save_model('./model/xgb_1000.model')
    print "Finished!"
    #xgb.plot_importance(bst)
    return preds


    
def del_category_data(data, features_type):
    category_index = [sample[0] for sample in features_type if sample[-1] == 'category']
    print("The number of category data is: {}".format(len(category_index)))
    for index in category_index:    
        del data[index]
    return data
    
def valid_test(train_x,train_label,test_x,test_label):
    #normalized_x = preprocessing.normalize(train_x)
    #normalized_test_data = preprocessing.normalize(test_x)
    xgb_clf(train_x,train_label,test_x, test_label)
    
def train():
    total = 15000
    train_num = 10000 
    test_num = total - train_num
    
    src_x = load_data("./data/train_x.csv")
    m,n = m,n = src_x.shape
    print("The demension of load src_x: m={},n={}".format(m,n))
    src_y = load_data("./data/train_y.csv")
    features_type = np.array(load_data("./data/features_type.csv"))    
    #src_x = del_category_data(src_x, features_type)
    m,n = src_x.shape
    print("The demension of after del src_x is: m={},n={}".format(m,n))    
    train_x = np.array(src_x)[0:train_num,1:n]
    train_id = np.array(src_x)[0:train_num,0]    
    #label
    label_y = np.array(src_y)[0:train_num,-1]
    label_id = np.array(src_y)[0:train_num,0]
    #test
    test_data = np.array(src_x)[train_num:total,1:n]
    test_id = np.array(src_x)[train_num:total,0]
    test_label = np.array(src_y)[train_num:total,-1]
    label_id = np.array(src_y)[train_num:total,0]
    print "********start_train*******"
    valid_test(train_x,label_y,test_data,test_label)
    
def test():
    print("Start detect....")
    src_test_data = load_data("./data/test_x.csv")
    features_type = np.array(load_data("./data/features_type.csv"))
    #src_test_data = del_category_data(src_test_data, features_type)
    test_data = np.array(src_test_data)[:,1:-1]
    test_id = np.array(src_test_data)[:,0]
    bst = xgb.Booster({'nthread':4})
    bst.load_model("./model/xgb_1000.model")
    #normalized_test_data = preprocessing.normalize(test_data)    
    dtest = xgb.DMatrix(test_data)
    pre = bst.predict(dtest)
    result = []
    for i in range(len(pre)):
        if pre[i] > 0.5:
            result.append(1)
        else:
            result.append(0)
    data = {'uid':test_id,'score':result}
    save_result = pd.DataFrame(data,columns = ['uid','score'])    
    save_result.to_csv('./result/xgb_result_1000.csv',index=False,encoding='utf-8')
    print("Finished!")
    
if __name__ == '__main__':
    train()
    #test()    
    

    
