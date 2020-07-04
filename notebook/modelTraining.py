# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 18:55:25 2020

@author: jayes
"""

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn .naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support
import pickle
import pandas as pd
import glob
from scipy import fftpack as fft
import pywt
import numpy as np
from dtw import dtw

def fastFourierTransform(cleaned_df):
    '''
    Performs Fast Fourier Transform on cleaned(pre-processed) data
    '''
    
    fastFourierTransform_df = fft.rfft(cleaned_df, n=5, axis=1)
    return pd.DataFrame(data=fastFourierTransform_df)

def dynamicTimeWarpDistance(df):
    '''
    Calculates Dynamic Time Warp Distance on cleaned(pre-processed) data
    '''
    manhattan_distance = lambda x, y: np.abs(x - y)
    dtw_distance = []
    for i in range(df.shape[0]-1):
      temp, _, _, _ = dtw(df.iloc[i,:52], df.iloc[i+1,:52], dist=manhattan_distance)
      dtw_distance.append(temp)
    dtw_distance.append(0)
    return pd.DataFrame(data=np.array(dtw_distance))

def discreteWaveletTransform(cleaned_df):
    '''
    Performs Discrete Wavelet Transform on cleaned(pre-processed) data
    '''
    
    cA, cD = pywt.dwt(cleaned_df, 'haar')
    return pd.DataFrame(cA),pd.DataFrame(cD)

def extractFeatures(df):
    
    #Universal Normalization
    cols1=['nose_x', 'nose_y','leftEye_x', 'leftEye_y','rightEye_x', 'rightEye_y','rightShoulder_x', 'rightShoulder_y', 'leftWrist_x',
       'leftWrist_y', 'rightWrist_x', 'rightWrist_y']
    
    cols2=['leftWrist_x', 'leftWrist_y', 'rightWrist_x', 'rightWrist_y']
    
    X=df[cols1]
    
    X['leftWrist_x']=(X['leftWrist_x']-X['nose_x'])/(X['leftEye_x']-X['rightEye_x'])
    X['leftWrist_y']=(X['leftWrist_y']-X['nose_y'])/(X['nose_y']-X['rightShoulder_y'])
    
    X['rightWrist_x']=(X['rightWrist_x']-X['nose_x'])/(X['leftEye_x']-X['rightEye_x'])
    X['rightWrist_y']=(X['rightWrist_y']-X['nose_y'])/(X['nose_y']-X['rightShoulder_y'])
    
    data=X[cols2]
    
    fft_df=fastFourierTransform(data)
    ca,cd=discreteWaveletTransform(data)
    
    features_data=pd.concat([X,fft_df,ca,cd],axis=1,ignore_index=True)
    
    labelValue=df['label']
    
    return features_data, labelValue
    
def kFoldCrossValidation(X,Y):

    kfold = model_selection.KFold(n_splits=10, random_state=42)
#    model = GradientBoostingClassifier(random_state=32)
#    model = LogisticRegression()   
#    model = GaussianNB()  
#    model = svm.SVC(gamma='auto')   
#    model = KNeighborsClassifier(n_neighbors=80, algorithm='auto' )  
#    model = RandomForestClassifier()   
#    model = DecisionTreeClassifier()
    model= ExtraTreesClassifier()   
    
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    print("Accuracy after KFold Cross Validation: %.3f%% " % (results.mean()*100.0)) 
    
def finalModelTestTrain(X,Y):
    
    #After kFoldCrossValidation select the below model that gives maximum accuracy
#    model = GradientBoostingClassifier(random_state=32)
#    model = LogisticRegression()   
#    model = GaussianNB()  
#    model = svm.SVC(gamma='auto')   
#    model = KNeighborsClassifier(n_neighbors=80, algorithm='auto' )  
#    model = RandomForestClassifier()   
#    model = DecisionTreeClassifier()
    model= ExtraTreesClassifier()

    test_size = 0.33
    seed = 42
    
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed, shuffle=True)
    
    # Fit the model on training set
    model.fit(X_train, Y_train)

    filename = 'finalModel.pkl'
    pickle.dump(model, open(filename, 'wb'))
     

    loaded_model = pickle.load(open(filename, 'rb'))
    y_pred=loaded_model.predict(X_test)
    

    
    result = loaded_model.score(X_test, Y_test)
    print("Accuracy of Trained Model: %.3f%% " % (result*100.0))
    print("Precison, Recall, F Socre:")
    print(precision_recall_fscore_support(Y_test, y_pred, average='macro'))

def generateModels(X,Y):
    
    model1 = KNeighborsClassifier(n_neighbors=80, algorithm='auto' ) 
    model1.fit(X, Y)
    filename = 'KNeighbors.pkl'
    pickle.dump(model1, open(filename, 'wb'))
    
    model2 = RandomForestClassifier() 
    model2.fit(X, Y)
    filename = 'RandomForest.pkl'
    pickle.dump(model2, open(filename, 'wb'))

    model3 = DecisionTreeClassifier()
    model3.fit(X, Y)
    filename = 'DecisionTree.pkl'
    pickle.dump(model3, open(filename, 'wb'))

    model4 = ExtraTreesClassifier()
    model4.fit(X, Y)
    filename = 'ExtraTrees.pkl'
    pickle.dump(model4, open(filename, 'wb'))
   
if __name__ == '__main__':
    
    df = pd.read_csv('FinalkeyPointsWithLabels.csv')
    features_data, labelValue= extractFeatures(df)
    
#    kFoldCrossValidation(features_data, labelValue)
    finalModelTestTrain(features_data, labelValue)
    generateModels(features_data, labelValue)