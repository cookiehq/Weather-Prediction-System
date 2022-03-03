# coding=gbk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO,StringIO

df = pd.read_csv('t_pm25.csv',encoding='utf-8')

#全省总览图
def chart_province():
    pd.DataFrame(df.groupby('city').AQI.mean().sort_values(ascending=False).tail(17)).plot.barh(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('全省空气质量总览')
    plt.xlabel('AQI')
    plt.ylabel('城市名')
    plt.xlim(50, 150)
    plt.legend('AQI')
    plt.grid(linestyle=':', color='w')

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#全省污染物热点图
def chart_province_pollute():
    # 全省主要污染物信息
    df_top10_polluted = []
    df_top10_polluted = pd.DataFrame(df_top10_polluted)
    for _ in range(0, 10):
        # 提取空气质量最严重的城市信息
        temp = df[df['city'] == df.groupby('city').AQI.mean().sort_values().tail(10).index[_]]
        df_top10_polluted = pd.concat([df_top10_polluted, temp])

    df_overpolluted = df_top10_polluted[df_top10_polluted['AQI'] >= 100][
        ['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']]

    plt.figure(figsize=(9.5, 6))
    sns.heatmap(df_overpolluted.corr(), vmax=1, square=False, annot=True, linewidth=1)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.yticks(rotation=0);
    plt.title('全省主要污染热点图')

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#全省不同时间段空气质量情况
def chart_province_time():
    pd.DataFrame(df.groupby('time_slot').AQI.mean().sort_values()).plot.line(figsize=(9.5, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('全省不同时间段空气质量情况')
    plt.xlabel('时间')
    plt.ylabel('AQI')
    # plt.xlim(70)
    plt.legend('AQI')
    plt.grid(linestyle=':', color='w')

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#当前城市的不同时间段空气质量情况
def chart_current_time():
    df_gd = df[df['place'] == '济南市（总）']
    pd.DataFrame(df_gd.groupby('time_slot').AQI.mean().sort_values()).plot.line(figsize=(9.5, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('济南不同时间段空气质量情况')
    plt.xlabel('时间段')
    plt.ylabel('AQI')
    plt.legend('AQI')
    plt.grid(linestyle=':', color='w')

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#获取所有城市名
def get_city():

    data = df.drop_duplicates(subset=['city'],keep='first')['city']
    data = data.astype(str)

    return data

#获取所有地区名
def get_place():

    data = df.drop_duplicates(subset=['place'],keep='first')['place']
    data = data.astype(str)

    return data

#当前城市的数据
def data_current():
    df['time'] = pd.to_datetime(df['time'])
    df_gd = df[df['place'] == '济南市（总）']
    data = df_gd[df['time'].dt.hour.isin(np.arange(12, 13))]
    data = data.astype(str)

    return data

'''
#当前城市的数据
def data_city():
    df['city'] = df['city'].astype('string')
    data = df.groupby('city').AQI.mean()
    data = data.astype(str)

    return data
'''

#某个城市的不同地区空气质量情况
def chart_city(city):

    df_gd = df[df['city'] == city]
    pd.DataFrame(df_gd.groupby('place').AQI.mean().sort_values()).plot.barh(figsize=(9.5, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(city + '不同地区空气质量情况')
    plt.xlabel('AQI')
    plt.ylabel('地区')
    #plt.xlim(80)
    plt.legend('AQI')
    plt.grid(linestyle=':', color='w')

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#某个城市的空气质量次数图
def chart_city_frequency(city):

    df_gd = df[df['city'] == city]
    df_gd.groupby('空气质量').time.count().plot.bar(figsize=(9.5, 6))
    plt.xticks(rotation=0)
    plt.ylabel('次数')
    plt.title(city + '空气质量次数图')

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#某个城市的主要污染图
def chart_city_pollutePM2_5(city):

    df_sz = df[df['city'] == city]
    df_sz_pollutant = df_sz[df_sz['AQI'] >= 100][['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']]
    sns.regplot(x='PM2_5', y='AQI', data=df_sz_pollutant);
    plt.title(city + '主要污染物')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd
#某个城市的主要污染图
def chart_city_pollutePM10(city):

    df_sz = df[df['city'] == city]
    df_sz_pollutant = df_sz[df_sz['AQI'] >= 100][['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']]
    sns.regplot(x='PM10', y='AQI', data=df_sz_pollutant);
    plt.title(city + '主要污染物')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#某城市污染物热点图
def chart_city_pollute(city):
    # 全省主要污染物信息
    df_gd = df[df['city'] == city]
    df_gd_pollutant = df_gd[df_gd['AQI'] >= 100][['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']]
    plt.figure(figsize=(9.5, 6))
    sns.heatmap(df_gd_pollutant.corr(), vmax=1, square=False, annot=True, linewidth=1)
    plt.yticks(rotation=0);
    plt.title(city + '主要污染物热点图')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#某个城市不同时间段空气质量情况
def chart_city_time(city):

    df_gd = df[df['city'] == city]
    pd.DataFrame(df_gd.groupby('time_slot').AQI.mean().sort_values()).plot.line(figsize=(9.5, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(city+'不同时间段空气质量情况')
    plt.xlabel('时间段')
    plt.ylabel('AQI')
    plt.legend('AQI')
    plt.grid(linestyle=':', color='w')

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#某个地区的空气质量次数图
def chart_place_frequency(place):

    df_gd = df[df['place'] == place]
    df_gd.groupby('空气质量').time.count().plot.barh(figsize=(9.5, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.xticks(rotation=0)
    plt.ylabel('次数')
    plt.title(place+'空气质量次数图')

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#某个地区不同时间段空气质量情况
def chart_place_time(place):

    df_gd = df[df['place'] == place]
    pd.DataFrame(df_gd.groupby('time_slot').AQI.mean().sort_values()).plot.line(figsize=(9.5, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(place+'不同时间段空气质量情况')
    plt.xlabel('时间段')
    plt.ylabel('AQI')
    plt.legend('AQI')
    plt.grid(linestyle=':', color='w')

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#某个地区的主要污染图
def chart_place_pollutePM2_5(place):

    df_sz = df[df['place'] == place]
    df_sz_pollutant = df_sz[df_sz['AQI'] >= 100][['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']]
    sns.regplot(x='PM2_5', y='AQI', data=df_sz_pollutant)
    plt.title(place + '主要污染物')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#某个地区的主要污染图
def chart_place_pollutePM10(place):

    df_sz = df[df['place'] == place]
    df_sz_pollutant = df_sz[df_sz['AQI'] >= 100][['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']]
    sns.regplot(x='PM10', y='AQI', data=df_sz_pollutant)
    plt.title(place + '主要污染物')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    return imd

#某个地区污染物热点图
def chart_place_pollute(place):
    # 全省主要污染物信息
    df_gd = df[df['place'] == place]
    df_gd_pollutant = df_gd[df_gd['AQI'] >= 100][['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']]
    plt.figure(figsize=(9.5, 6))
    sns.heatmap(df_gd_pollutant.corr(), vmax=1, square=False, annot=True, linewidth=1)
    plt.yticks(rotation=0);
    plt.title(place+'主要污染物热点图')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd




