from flask import Flask, request
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from flask import Flask,request,redirect,jsonify
import os
import urllib.request
from werkzeug.utils import secure_filename
from base64 import b64encode
import json
import csv
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils import shuffle
import pickle
import glob
from scipy import fftpack as fft
import pywt
from dtw import dtw

UPLOAD_FOLDER = '/upload'
# ALLOWED_EXTENSIONS = {'json'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key="keyy"

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

def fastFourierTransform(cleaned_df):
    '''
    Performs Fast Fourier Transform on cleaned(pre-processed) data
    '''
    
    fastFourierTransform_df = fft.rfft(cleaned_df, n=5, axis=1)
    return pd.DataFrame(data=fastFourierTransform_df)

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
    
    return features_data

def predictFinalLabel(y_pred):
    
    unq,pos = np.unique(y_pred,return_inverse=True)
    cnts = np.bincount(pos)
    maxp = cnts.argmax()
    
    return str(unq[maxp])
    
@app.route("/")
def hello_world():
    return "MC Assignment 2"

@app.route("/getLabels", methods=['POST','PUT'])
def getLabels():
    
    request_data = request.get_json()

    filename='keyPts.json'
    with open(filename, 'w') as f:
        json.dump(request_data, f)
    info = json.loads(open(filename).read())

    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
       'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
       'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
       'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
       'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
       'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
       'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
       'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
       'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
    # info = json.loads(open('key_points1.json', 'r').read())  #Put the name of the .json here
    csv = np.zeros((len(info), len(columns)))
    for i in range(csv.shape[0]):
        one = []
        one.append(info[i]['score'])
        for object in info[i]['keypoints']:
            one.append(object['score'])
            one.append(object['position']['x'])
            one.append(object['position']['y'])
        csv[i] = np.array(one)
    df=pd.DataFrame(csv, columns=columns) #dataframe
    
    features_df= extractFeatures(df)
    
    loaded_model1 = pickle.load(open('knn.pkl', 'rb'))
    y_pred_KNeighbors=loaded_model1.predict(features_df)
    label1=predictFinalLabel(y_pred_KNeighbors)
    
    loaded_model2 = pickle.load(open('RandomForest.pkl', 'rb'))
    y_pred_RandomForest=loaded_model2.predict(features_df)
    label2=predictFinalLabel(y_pred_RandomForest)
    
    loaded_model3 = pickle.load(open('DecisionTree.pkl', 'rb'))
    y_pred_DecisionTree=loaded_model3.predict(features_df)
    label3=predictFinalLabel(y_pred_DecisionTree)
    
    loaded_model4 = pickle.load(open('ExtraTrees.pkl', 'rb'))
    y_pred_ExtraTrees=loaded_model4.predict(features_df)
    label4=predictFinalLabel(y_pred_ExtraTrees)
    
    jsonResponse = {
            '1': label1,
            '2': label2,
            '3': label3,
            '4': label4
        }
    return jsonify(jsonResponse)


if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000)