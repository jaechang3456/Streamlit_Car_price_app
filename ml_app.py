import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import pickle
import os
import tensorflow as tf
import h5py
import joblib

def run_ml_app() :
    st.subheader('Machine Learning')

    # 1. 유저한테 입력을 받는다.
    # 1-1. 성별 입력
    gender = st.radio('성별을 선택하세요.', ['남자','여자'])
    if gender == '남자' :
        gender = 1
    elif gender =='여자' :
        gender = 0
    # st.write(gender)

    # 1-2. 나이 입력
    age = st.number_input('나이를 입력하세요.', min_value=0, max_value=120)

    # 1-3. 연봉 입력
    salary = st.number_input('연봉을 입력하세요.', min_value=0)

    # 1-4. 빚 입력
    debt = st.number_input('빚을 입력하세요.', min_value=0)

    # 1-5. 자산 입력
    worth = st.number_input('자산을 입력하세요.',min_value=0)

    # 2. 예측한다.
    # 2-1. 모델 불러오기
    model = tensorflow.keras.models.load_model('data/Car_Ai.h5')

    # 2-2. 예측을 위해 입력받은 데이터 가공 (모양 바꾸기 및 피쳐 스케일링)
    new_data = np.array( [gender, age, salary, debt, worth ] )
    new_data = new_data.reshape(1,-1)
    sc_X = joblib.load('data/sc_X.pkl')
    new_data = sc_X.transform(new_data)

    # 2-3. 예측 한다.
    y_pred = model.predict(new_data)

    # 2-4. 예측 결과는 스케일 된 결과이므로, 다시 원 값으로 돌린다.
    sc_y = joblib.load('data/sc_y.pkl')
    y_pred_original = sc_y.inverse_transform(y_pred)

    # 3. 예측 결과를 화면에 보여준다.
    btn = st.button('결과 보기')
    if btn :
        st.write('예측 결과입니다. {:,.1f}달러의 차를 살 수 있습니다.'.format(y_pred_original[0,0]))
