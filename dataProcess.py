import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyecharts.charts import Geo
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import forecast

df = pd.read_csv('t_pm25.csv',encoding='utf-8')

#df['city'] = df['city'].astype('string')

#print(df.head())
#print(df.info())
#print(df.isnull().any())
#print(df.shape)

#print('数据中的城市数量为：', df.city.value_counts().count())

#数据清洗

'''
for city in df.place.value_counts().index:

    df.loc[(df['place'] == city) & (df['AQI'].isnull()), 'AQI'] = df[df['place'] == city]['AQI'].mean()
    df.loc[(df['place'] == city) & (df['PM2_5'].isnull()), 'PM2_5'] = df[df['place'] == city]['PM2_5'].mean()
    df.loc[(df['place'] == city) & (df['PM10'].isnull()), 'PM10'] = df[df['place'] == city]['PM10'].mean()
    df.loc[(df['place'] == city) & (df['CO'].isnull()), 'CO'] = df[df['place'] == city]['CO'].mean()
    df.loc[(df['place'] == city) & (df['SO2'].isnull()), 'SO2'] = df[df['place'] == city]['SO2'].mean()
    df.loc[(df['place'] == city) & (df['O3'].isnull()), 'O3'] = df[df['place'] == city]['O3'].mean()
    df.loc[(df['place'] == city) & (df['O3'].isnull()), 'O3'] = df[df['place'] == city]['O3'].mean()
    df.loc[(df['place'] == city) & (df['NO2'].isnull()), 'NO2'] = df[df['place'] == city]['NO2'].mean()

#df.drop(['f_id'], axis=1, inplace=True)
#df.drop(['f_arcacode'], axis=1, inplace=True)
#df.drop(['AQItype'], axis=1, inplace=True)
#df.drop(['f_O3per8h'], axis=1, inplace=True)
#df.drop(['f_majorpollutant'], axis=1, inplace=True)

#print(df.info())

df['time'] = pd.to_datetime(df['time'])

df.drop(['AQItype'], axis=1, inplace=True)
df.drop(['major_pollutant'], axis=1, inplace=True)

bin_edges = [0, 50, 100, 150, 200, 300, 1210] # 根据AQI的划分等级设置标签
bin_names = ['优级', '良好', '轻度污染', '中度污染', '重度污染', '重污染']
df['空气质量'] = pd.cut(df['AQI'], bin_edges, labels=bin_names)

# 根据时间创建时间段列
time_slot = {0: '晚上',
           1: '凌晨',
           2: '凌晨',
           3: '凌晨',
           4: '凌晨',
           5: '凌晨',
           6: '凌晨',
           7: '上午',
           8: '上午',
           9: '上午',
           10: '上午',
           11: '上午',
           12: '上午',
           13: '下午',
           14: '下午',
           15: '下午',
           16: '下午',
           17: '下午',
           18: '下午',
           19: '晚上',
           20: '晚上',
           21: '晚上',
           22: '晚上',
           23: '晚上'

}
#df['time'] = pd.to_datetime(df['time'])
#df['time_slot'] = df['time'].apply(lambda x : time_slot [x.hour])

time_slot = {0: '晚上',1: '凌晨',2: '凌晨',3: '凌晨',4: '凌晨',5: '凌晨',6: '凌晨',7: '上午',8: '上午',9: '上午',10: '上午',11: '上午',
12: '上午',13: '下午',14: '下午',15: '下午',16: '下午',17: '下午',18: '下午',19: '晚上',20: '晚上',21: '晚上',22: '晚上',23: '晚上'}
#df['time'] = pd.to_datetime(df['time'])
#df['time_slot'] = df['time'].apply(lambda x : time_slot [x.hour])
'''

#df.info() #查看time列的数据类型

#print(df.head())
#print(df.info())

#保存csv
#df.to_csv('t_pm25.csv',encoding='utf-8',index=False)



#预测结果绘图
'''
df['time'] = pd.to_datetime(df['time'])
df_gd = df[df['place'] == '济南市（总）']
data = df_gd[df['time'].dt.hour.isin(np.arange(12, 13))]

data.groupby('time').AQI.mean().plot.line(figsize=(10,5))
plt.rcParams['font.sans-serif']=['SimHei']#显示中文标签
plt.rcParams['axes.unicode_minus']=False
plt.ylabel('AQI')
plt.xlabel('时间')
plt.title('济南AQI随时间变化图')

plt.show();

df['time'] = pd.to_datetime(df['time'])
df_gd = df[df['place'] == '济南市（总）']
data = df_gd[df['time'].dt.hour.isin(np.arange(12, 13))]
time = data['time']
AQI = data['AQI']
#data = data.astype(str)


X = df.drop(['city','time','place','空气质量','time_slot'],axis=1)
y = df['AQI']
# 将数据划分为数据集与测试集
# test_size:测试集大小
# random_state：随机种子，用来产生相同的随机数系列
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0)
lr = LinearRegression()
# 使用训练集，训练模型
lr.fit(X_train, y_train)
print("权重：", lr.coef_)
print("截距：", lr.intercept_)


# 从训练集学习到的模型参数（W与b），确定方程，就可以进行预测
y_hat = lr.predict(X_test)
print("实际值：", y_test[:5].values)
print("预测值：", y_hat[:5])
# score其实求解的就是r^2的值
print("训练集R^2：",lr.score(X_train, y_train))
print("测试集R^2：",lr.score(X_test, y_test))

# 绘制预测结果图
plt.figure(figsize=(15, 8))
plt.plot(y_test.values, label="真实值", color="indianred", marker="o")
plt.plot(y_hat, label="预测值", color="c", marker="*")
plt.xlabel("测试集数据序列")
plt.ylabel("数据值")
plt.title("线性回归预测结果", fontsize=12)
plt.legend(loc=1)
#plt.show()

'''
'''
df['time'] = pd.to_datetime(df['time'])
df_gd = df[df['place'] == '济南市（总）']
X = df_gd['time']
y = df_gd['AQI']

print(X)

df['time'] = pd.to_datetime(df['time'])
df_gd = df[df['place'] == '济南市（总）']
data = df_gd[df['time'].dt.hour.isin(np.arange(12, 13))]



df = df.drop(['city','time','place','空气质量','time_slot','PM10','CO','NO2','O3','SO2'],axis=1)

dataset = df.values
# 将整型变为float
dataset = dataset.astype('float32')
#归一化 在下一步会讲解
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.65)
trainlist = dataset[:train_size]
testlist = dataset[train_size:]

def create_dataset(dataset, look_back):
#这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX),np.array(dataY)
#训练数据太少 look_back并不能过大
look_back = 1
trainX,trainY  = create_dataset(trainlist,look_back)
testX,testY = create_dataset(testlist,look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1] ,1 ))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(None,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
model.save(os.path.join("DATA","Test" + ".h5"))

#model = load_model(os.path.join("DATA","Test" + ".h5"))
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#反归一化
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

plt.plot(trainY)
plt.plot(trainPredict[1:])
plt.show()
plt.plot(testY)
plt.plot(testPredict[1:])
plt.show()

'''

















