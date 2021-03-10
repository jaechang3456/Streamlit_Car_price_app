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

from eda_app import run_eda_app
from ml_app import run_ml_app

def main():

    st.title('자동차 가격 예측')

    # 사이드바 메뉴
    menu = ['Home','EDA','ML']
    choice = st.sidebar.selectbox('Menu',menu)

    if choice == 'Home':
        st.write('이 앱은 고객 데이터와 자동차 구매 데이터에 대한 내용입니다. 해당 고객의 정보를 입력하면, 얼마정도의 차를 구매할 수 있는지를 예측하는 앱입니다.')       
        st.write('왼쪽의 사이드바에서 선택하세요.')

    elif choice == 'EDA' :
        run_eda_app()
    elif choice == 'ML' :
        run_ml_app()

#         st.subheader('연봉이 가장 높은사람')
#         st.dataframe(df.loc[df['Annual Salary'] == df['Annual Salary'].max(), ])

#         st.subheader('나이가 가징 어린 고객의 연봉 확인')
#         st.dataframe(df.loc[df['Age'] == df['Age'].min() , ])

#         elif option == '상관관계 분석':

#             df = pd.read_csv('data/Car_Purchasing_Data.csv')
#             st.subheader('상관관계 분석을 위한 pairplot그리기')
#             st.pyplot(sns.pairplot(df))
#             st.dataframe(df.corr())

#         elif option == '학습을 위한 데이터 가공 및 학습':

#             df = pd.read_csv('data/Car_Purchasing_Data.csv')
#             st.subheader('NaN값이 있는지 확인하고, 있을경우 해결하기')
#             st.write(df.isna().sum())

#             st.subheader('학습을 위한 X,y만들기')
#             X = df.iloc[ : , 3:7+1]
#             st.write('X')
#             st.dataframe(X)
#             y = df['Car Purchase Amount']
#             st.write('y')
#             st.dataframe(y)

#             # st.subheader('피쳐 스케일링 하기')
#             # sc_X = MinMaxScaler()
#             # X_scaled = sc_X.fit_transform(X)
#             # st.write('X_scaled')
#             # st.write(X_scaled)
#             # st.write('X_scaled의 모양 : {}'.format(X_scaled.shape))
#             st.write('X의 모양 : {}'.format(X.shape))

#             st.subheader('학습을 위해, y의 shape 변경하기')
#             st.write('y의 모양 : {}'.format(y.shape))
#             y = y.values.reshape(-1,1)
#             st.write('y')
#             st.write(y)
#             st.write('y의 변경된 모양 : {}'.format(y.shape))

#             # st.subheader('y도 피쳐 스케일링 하기')
#             # sc_y = MinMaxScaler()
#             # y_scaled = sc_y.fit_transform(y)
#             # st.write('y_scaled')
#             # st.write(y_scaled)

#             st.write('트레이닝 셋과 테스트 셋으로 분리하기(테스트 사이즈 25%, 랜덤스테이트 50)')
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 50)

#             st.subheader('딥러닝을 이용한 모델링 하기')
#             model = Sequential()
#             model.add( Dense( input_dim = 5, units = 25, activation='relu' ))
#             model.add( Dense( units= 40, activation='relu' ))
#             model.add( Dense( units= 25, activation='relu' ))
#             model.add( Dense( units = 1, activation='linear'))
#             mdsm = model.summary()
#             st.write(mdsm)

#             st.write("옵티마이저 'adam', 로스펑션 'mean_squared_error' 셋팅하여 컴파일하기")
#             model.compile(loss = 'mean_squared_error', optimizer='adam')

#             # st.write('학습 진행하기')
#             # history = model.fit(X_train, y_train, epochs=20, batch_size = 25, verbose=1)
#             # st.write(history)

#             TYPE='type'
#             model_type='mobilenetv2'
#             user='block'
#             iteration='1-2'

#             first_time_training=True
#             PROJECT_PATH= 'data'
#             HDF5_DATASET_PATH=PROJECT_PATH+'/car_datasets/car-type-dataset-SIZE224-train-dev-test-v2.hdf5'
#             TARGET_CLASSIFICATION_MODEL=PROJECT_PATH+'/trained-models/'+model_type+'/'+'car-classification-by-'+TYPE+'-'+model_type+'-'+user+'-'+iteration+'.h5'
#             CHECKPOINT_PATH = PROJECT_PATH+'/checkpoints/'+model_type+'/'+'by-'+TYPE+'-'+model_type+'-'+user+'-'+iteration+'.h5'
#             LOGFILE_PATH=PROJECT_PATH+'/log/'+model_type+'/'+model_type+'-by-'+TYPE+'-training-log'+user+'-'+iteration+'.csv'

#             st.write('콜백만들기 : 가장 좋은 모델을 자동 저장, 에포크 로그도 저장')
#             cp = ModelCheckpoint(filepath=CHECKPOINT_PATH, monitor='val_loss', mode='min',save_best_only=True, verbose=1)
#             csv_logger = CSVLogger(filename=LOGFILE_PATH, append=True)

#             if not os.path.exists(PROJECT_PATH + '/checkpoints/' + model_type + '/' ) :
#                 os.makedirs(PROJECT_PATH + '/checkpoints/' + model_type + '/')

#             if not os.path.exists(PROJECT_PATH + '/log/' + model_type + '/' ) :
#                 os.makedirs(PROJECT_PATH + '/log/' + model_type + '/')

#             st.write('학습 진행하면서 모델 저장하기')
#             history = model.fit(X_train, y_train, validation_data=(X_test, y_test) ,epochs=50,callbacks=[cp, csv_logger])

#             st.subheader('테스트셋으로 예측 해보기')
#             y_pred = model.predict(X_test)
#             ret_df = pd.DataFrame( {'실제값' : y_test.reshape(-1, ) , '예측값' : y_pred.reshape(-1, ), '오차' : y_pred.reshape(-1,)-y_test.reshape(-1,) } )
#             st.write(ret_df)

#             st.subheader('실제값과 예측값 시각화하기')
#             st.line_chart(y_test)
#             st.line_chart(y_pred)

#     elif choice == '새로운 데이터 예측하기':
#         st.subheader('저장한 모델을 불러와서 새로운 데이터 예측하기')
#         gender = st.text_input('성별입력(남/여)')
#         if gender == '남':
#             gender = 1
#         else :
#             gender = 0
#         #st.write(gender)
#         age = st.number_input('나이 입력')
#         year = st.number_input('연봉 입력')
#         dept = st.number_input('신용카드 빚 입력')
#         money = st.number_input('재산 입력')
#         new_X = np.array( [gender, age, year, dept, money] )
#         new_X = new_X.reshape(1, -1)


#         TYPE='type'
#         model_type='mobilenetv2'
#         user='block'
#         iteration='1-2'

#         first_time_training=True
#         PROJECT_PATH= 'data'
#         HDF5_DATASET_PATH=PROJECT_PATH+'/car_datasets/car-type-dataset-SIZE224-train-dev-test-v2.hdf5'
#         TARGET_CLASSIFICATION_MODEL=PROJECT_PATH+'/trained-models/'+model_type+'/'+'car-classification-by-'+TYPE+'-'+model_type+'-'+user+'-'+iteration+'.h5'
#         CHECKPOINT_PATH = PROJECT_PATH+'/checkpoints/'+model_type+'/'+'by-'+TYPE+'-'+model_type+'-'+user+'-'+iteration+'.h5'
#         LOGFILE_PATH=PROJECT_PATH+'/log/'+model_type+'/'+model_type+'-by-'+TYPE+'-training-log'+user+'-'+iteration+'.csv'
#         model = tf.keras.models.load_model(CHECKPOINT_PATH)
#         new_y_pred = model.predict(new_X)
#         st.write('예상 자동차 구매 가격은{} 입니다.'.format(new_y_pred))

if __name__ == '__main__':
    main()