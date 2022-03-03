# coding=gbk
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import base64
from io import BytesIO,StringIO

df = pd.read_csv('t_pm25.csv',encoding='utf-8')

#ȫʡ����ͼ
def chart_province():
    pd.DataFrame(df.groupby('city').AQI.mean().sort_values(ascending=False).tail(17)).plot.barh(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ��ʾ���ı�ǩ
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('ȫʡ������������')
    plt.xlabel('AQI')
    plt.ylabel('������')
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

#ȫʡ��Ⱦ���ȵ�ͼ
def chart_province_pollute():
    # ȫʡ��Ҫ��Ⱦ����Ϣ
    df_top10_polluted = []
    df_top10_polluted = pd.DataFrame(df_top10_polluted)
    for _ in range(0, 10):
        # ��ȡ�������������صĳ�����Ϣ
        temp = df[df['city'] == df.groupby('city').AQI.mean().sort_values().tail(10).index[_]]
        df_top10_polluted = pd.concat([df_top10_polluted, temp])

    df_overpolluted = df_top10_polluted[df_top10_polluted['AQI'] >= 100][
        ['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']]

    plt.figure(figsize=(9.5, 6))
    sns.heatmap(df_overpolluted.corr(), vmax=1, square=False, annot=True, linewidth=1)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ��ʾ���ı�ǩ
    plt.rcParams['axes.unicode_minus'] = False
    plt.yticks(rotation=0);
    plt.title('ȫʡ��Ҫ��Ⱦ�ȵ�ͼ')

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#ȫʡ��ͬʱ��ο����������
def chart_province_time():
    pd.DataFrame(df.groupby('time_slot').AQI.mean().sort_values()).plot.line(figsize=(9.5, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ��ʾ���ı�ǩ
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('ȫʡ��ͬʱ��ο����������')
    plt.xlabel('ʱ��')
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

#��ǰ���еĲ�ͬʱ��ο����������
def chart_current_time():
    df_gd = df[df['place'] == '�����У��ܣ�']
    pd.DataFrame(df_gd.groupby('time_slot').AQI.mean().sort_values()).plot.line(figsize=(9.5, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('���ϲ�ͬʱ��ο����������')
    plt.xlabel('ʱ���')
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

#��ȡ���г�����
def get_city():

    data = df.drop_duplicates(subset=['city'],keep='first')['city']
    data = data.astype(str)

    return data

#��ȡ���е�����
def get_place():

    data = df.drop_duplicates(subset=['place'],keep='first')['place']
    data = data.astype(str)

    return data

#��ǰ���е�����
def data_current():
    df['time'] = pd.to_datetime(df['time'])
    df_gd = df[df['place'] == '�����У��ܣ�']
    data = df_gd[df['time'].dt.hour.isin(np.arange(12, 13))]
    data = data.astype(str)

    return data

'''
#��ǰ���е�����
def data_city():
    df['city'] = df['city'].astype('string')
    data = df.groupby('city').AQI.mean()
    data = data.astype(str)

    return data
'''

#ĳ�����еĲ�ͬ���������������
def chart_city(city):

    df_gd = df[df['city'] == city]
    pd.DataFrame(df_gd.groupby('place').AQI.mean().sort_values()).plot.barh(figsize=(9.5, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ��ʾ���ı�ǩ
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(city + '��ͬ���������������')
    plt.xlabel('AQI')
    plt.ylabel('����')
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

#ĳ�����еĿ�����������ͼ
def chart_city_frequency(city):

    df_gd = df[df['city'] == city]
    df_gd.groupby('��������').time.count().plot.bar(figsize=(9.5, 6))
    plt.xticks(rotation=0)
    plt.ylabel('����')
    plt.title(city + '������������ͼ')

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#ĳ�����е���Ҫ��Ⱦͼ
def chart_city_pollutePM2_5(city):

    df_sz = df[df['city'] == city]
    df_sz_pollutant = df_sz[df_sz['AQI'] >= 100][['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']]
    sns.regplot(x='PM2_5', y='AQI', data=df_sz_pollutant);
    plt.title(city + '��Ҫ��Ⱦ��')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ��ʾ���ı�ǩ
    plt.rcParams['axes.unicode_minus'] = False

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd
#ĳ�����е���Ҫ��Ⱦͼ
def chart_city_pollutePM10(city):

    df_sz = df[df['city'] == city]
    df_sz_pollutant = df_sz[df_sz['AQI'] >= 100][['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']]
    sns.regplot(x='PM10', y='AQI', data=df_sz_pollutant);
    plt.title(city + '��Ҫ��Ⱦ��')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ��ʾ���ı�ǩ
    plt.rcParams['axes.unicode_minus'] = False

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#ĳ������Ⱦ���ȵ�ͼ
def chart_city_pollute(city):
    # ȫʡ��Ҫ��Ⱦ����Ϣ
    df_gd = df[df['city'] == city]
    df_gd_pollutant = df_gd[df_gd['AQI'] >= 100][['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']]
    plt.figure(figsize=(9.5, 6))
    sns.heatmap(df_gd_pollutant.corr(), vmax=1, square=False, annot=True, linewidth=1)
    plt.yticks(rotation=0);
    plt.title(city + '��Ҫ��Ⱦ���ȵ�ͼ')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ��ʾ���ı�ǩ
    plt.rcParams['axes.unicode_minus'] = False

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#ĳ�����в�ͬʱ��ο����������
def chart_city_time(city):

    df_gd = df[df['city'] == city]
    pd.DataFrame(df_gd.groupby('time_slot').AQI.mean().sort_values()).plot.line(figsize=(9.5, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ��ʾ���ı�ǩ
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(city+'��ͬʱ��ο����������')
    plt.xlabel('ʱ���')
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

#ĳ�������Ŀ�����������ͼ
def chart_place_frequency(place):

    df_gd = df[df['place'] == place]
    df_gd.groupby('��������').time.count().plot.barh(figsize=(9.5, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ��ʾ���ı�ǩ
    plt.rcParams['axes.unicode_minus'] = False
    plt.xticks(rotation=0)
    plt.ylabel('����')
    plt.title(place+'������������ͼ')

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#ĳ��������ͬʱ��ο����������
def chart_place_time(place):

    df_gd = df[df['place'] == place]
    pd.DataFrame(df_gd.groupby('time_slot').AQI.mean().sort_values()).plot.line(figsize=(9.5, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ��ʾ���ı�ǩ
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(place+'��ͬʱ��ο����������')
    plt.xlabel('ʱ���')
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

#ĳ����������Ҫ��Ⱦͼ
def chart_place_pollutePM2_5(place):

    df_sz = df[df['place'] == place]
    df_sz_pollutant = df_sz[df_sz['AQI'] >= 100][['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']]
    sns.regplot(x='PM2_5', y='AQI', data=df_sz_pollutant)
    plt.title(place + '��Ҫ��Ⱦ��')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ��ʾ���ı�ǩ
    plt.rcParams['axes.unicode_minus'] = False

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd

#ĳ����������Ҫ��Ⱦͼ
def chart_place_pollutePM10(place):

    df_sz = df[df['place'] == place]
    df_sz_pollutant = df_sz[df_sz['AQI'] >= 100][['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']]
    sns.regplot(x='PM10', y='AQI', data=df_sz_pollutant)
    plt.title(place + '��Ҫ��Ⱦ��')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ��ʾ���ı�ǩ
    plt.rcParams['axes.unicode_minus'] = False

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    return imd

#ĳ��������Ⱦ���ȵ�ͼ
def chart_place_pollute(place):
    # ȫʡ��Ҫ��Ⱦ����Ϣ
    df_gd = df[df['place'] == place]
    df_gd_pollutant = df_gd[df_gd['AQI'] >= 100][['AQI', 'PM2_5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']]
    plt.figure(figsize=(9.5, 6))
    sns.heatmap(df_gd_pollutant.corr(), vmax=1, square=False, annot=True, linewidth=1)
    plt.yticks(rotation=0);
    plt.title(place+'��Ҫ��Ⱦ���ȵ�ͼ')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ��ʾ���ı�ǩ
    plt.rcParams['axes.unicode_minus'] = False

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims

    plt.clf()

    return imd




