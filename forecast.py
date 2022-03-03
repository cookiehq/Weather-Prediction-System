# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
import base64
from io import BytesIO,StringIO

df = pd.read_csv('t_pm25.csv',encoding='utf-8')

def create_dataset(dataset,look_back = 100):
    dataX,dataY = [],[]
    for i in range(len(dataset)-look_back):    #  i 从 0 ：3548 ，dataX :0-100，1-101 ,dataY :100,101,102..
        a = dataset[i:(i+look_back),0]
        dataX.append(a)
        dataY.append(dataset[i+look_back,0])
    return np.array(dataX),np.array(dataY)

def get_forecast():

    df['time'] = pd.to_datetime(df['time'])
    df_gd = df[df['place'] == '济南市（总）']
    data = df_gd[df['time'].dt.hour.isin(np.arange(8, 10))]
    AQI = data.AQI
    dataset = AQI.values


    # 设置随机种子
    np.random.seed(7)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # 数据集归一化
    dataset = scaler.fit_transform(dataset.reshape(-1, 1))
    train = dataset

    # 设置时间滑窗，创建训练集
    look_back = 100
    trainX, trainY = create_dataset(train, look_back)
    # train = dataset

    # reshape trainX 成LSTM可以接受的输入 (样本，时间步，特征)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))

    # 搭建lstm网络
    model = Sequential()
    model.add(LSTM(1, input_shape=(1, look_back)))
    # 输出节点为1，输入的每个样本的长度为look_back
    model.add(Dense(1))  # 一个全连接层
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # 预测训练集
    trainPredict = model.predict(trainX)

    testx = [0.] * (7 + look_back)  # testx 为107个数据  列表
    testx[0:look_back] = train[-look_back:]  # testx 0：100个数据是 train(dataset) 倒数100个
    testx = K.cast_to_floatx(testx)
    testx = np.array(testx)  # 把testx变为一个数组 ，二维(1,107)
    testPredict = [0.] * 7  # testPredict 是一个 7 个数据的列表
    for i in range(7):
        Xtest = testx[-look_back:]  # Xtest 是testx -100到最后的数据，共100个
        Xtest = np.reshape(Xtest, (1, 1, look_back))
        testy = model.predict(Xtest)
        testx[look_back + i] = testy
        testPredict[i] = testy
    testPredict = np.array(testPredict)
    testPredict = np.reshape(testPredict, (7, 1))

    # 反标准化
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)

    # 输出RMSE
    trainScore = math.sqrt(mean_squared_error(trainY[0, :], trainPredict[:, 0]))
    print('Train Score :%.2f RMSE' % (trainScore))

    trainPredictPlot = np.reshape(np.array([None] * (len(dataset) + 7)), ((len(dataset) + 7), 1))
    trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict

    testPredictPlot = np.reshape(np.array([None] * (len(dataset) + 7)), ((len(dataset) + 7), 1))
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(dataset):(len(dataset) + 7), :] = testPredict

    plt.plot(scaler.inverse_transform(dataset), label='true')
    plt.plot(trainPredictPlot, label='trainpredict')
    plt.plot(testPredictPlot, label='testpredict')
    plt.legend()  # 给图像加上图例

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd
