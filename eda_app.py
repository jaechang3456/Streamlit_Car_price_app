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

    # 컬럼을 하나만 선택하면, 해당 컬럼의 최대,최소값에 해당하는 사람의 데이터를 화면에 보여주는 기능

    number_columns = corr_col_list
    selected_col = st.selectbox('최대,최소값 확인할 컬럼 선택',number_columns)

    min_data = car_df[selected_col].min() == car_df[selected_col]
    st.write('최소값 데이터')
    st.dataframe(car_df.loc[ min_data, ] )

    max_data = car_df[selected_col].max() == car_df[selected_col]
    st.write('최대값 데이터')
    st.dataframe(car_df.loc[ max_data, ] )

    # 고객이름을 검색할 수 있는 기능 개발
    # 1. 유저한테 검색어를 받자
    name = st.text_input('고객의 이름을 입력하세요.')
    # 2. 검색어를 데이터프레임 커스터머네임 컬럼에 검색하자.
    search_name = car_df.loc[car_df['Customer Name'].str.contains(name, case=False) , ]
    # 3. 화면에 결과를 보여주자.
    if len(search_name) != 0 :
        st.dataframe(search_name)
    else :
        st.write('존재하지 않는 고객입니다.')
    