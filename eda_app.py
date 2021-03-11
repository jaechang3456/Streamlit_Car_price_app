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

def run_eda_app() :
    st.subheader('EDA 화면입니다.')

    car_df = pd.read_csv('data/Car_Purchasing_Data.csv', encoding='ISO-8859-1')

    radio_menu = ['DataFrame', 'Statistics']
    selected_radio = st.radio('선택하세요.',radio_menu)

    if selected_radio == 'DataFrame':
        st.dataframe(car_df)
    elif selected_radio == 'Statistics' :
        st.dataframe(car_df.describe())

    col_list = list(car_df.columns)
    # st.write(col_list)
    selected_col_list = st.multiselect('데이터를 확인 할 컬럼을 선택하세요.',col_list)

    if len(selected_col_list) != 0 :
        st.dataframe(car_df[selected_col_list])
    else : 
        st.write('선택한 컬럼이 없습니다.')

    # 멀티셀렉트에 컬럼명 보이고 해당 컬럼들에 대한 상관관계 보이기.
    # 단, 컬럼들은 숫자 컬럼들만 멀티셀렉트에 나타나야 합니다.

    car_df.dtypes != object
    corr_col_list = car_df.columns[car_df.dtypes != object ]
    # st.write(corr_col_list)
    selected_corr = st.multiselect('상관 계수를 볼 컬럼을 선택하세요',corr_col_list)
    
    if len(selected_corr) != 0  :
        st.dataframe(car_df[selected_corr].corr())
        # 위에서 선택한 컬럼들을 이용해서, 시본의 페어플롯을 그린다.
        fig = sns.pairplot(car_df[selected_corr])
        st.pyplot(fig)

    else :
        st.write('선택한 컬럼이 없습니다.')
