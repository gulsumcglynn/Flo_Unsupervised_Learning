###############################################################
# Gözetimsiz Öğrenme ile Müşteri Segmentasyonu (Customer Segmentation with Unsupervised Learning)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################

# Unsupervised Learning yöntemleriyle (Kmeans, Hierarchical Clustering )  müşteriler kümelere ayrılıp ve davranışları gözlemlenmek istenmektedir.

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# 20.000 gözlem, 13 değişken

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
# store_type : 3 farklı companyi ifade eder. A company'sinden alışveriş yapan kişi B'dende yaptı ise A,B şeklinde yazılmıştır.
###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Hazırlama
           # 1. flo_data_20K.csv.csv verisini okuyunuz.
           # 2. Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz. Tenure(Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.

# GÖREV 2: K-Means ile Müşteri Segmentasyonu
           # 1. Değişkenleri standartlaştırınız.
           # 2. Optimum küme sayısını belirleyiniz.
           # 3. Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
           # 4. Herbir segmenti istatistiksel olarak inceleyeniz.

# GÖREV 3: Hierarchical Clustering ile Müşteri Segmentasyonu
           # 1. Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
           # 2. Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
           # 3. Herbir segmenti istatistiksel olarak inceleyeniz.


#Görev 1: Veriyi Hazırlama

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import seaborn as sns
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
import warnings
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#Görev 1: Veriyi Hazırlama

#Adım 1: flo_data_20K.csv verisini okutunuz.
df = pd.read_csv("datasets/flo_data_20k.csv")
df.head()
df.info()
df.isnull().sum()
#Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
#Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz

last_date = df["last_order_date"].max()
last_date = datetime.strptime(last_date, "%Y-%m-%d")
last_date = last_date + timedelta(days=1)  # last date 31.5.2021
analysis_date = datetime.strftime(last_date, "%Y-%m-%d")
analysis_date = pd.to_datetime(analysis_date)
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df["recency"] = (analysis_date - df["last_order_date"]).astype('timedelta64[D]')
df["tenure"] = (df["last_order_date"] - df["first_order_date"]).astype('timedelta64[D]')
# müşteri segmentasyonuna geçmeden genel veri setiyle ilgili analizlerimize bakalım.

def grab_col_names(dataframe,cat_th=10,car_th=20):
    cat_cols=[col for col in dataframe.columns if dataframe[col].dtypes=="object"]
    num_but_cat=[col for col in dataframe.columns if dataframe[col].nunique()<cat_th and dataframe[col].dtypes!="object"]
    cat_but_car=[col for col in dataframe.columns if dataframe[col].nunique()>car_th and dataframe[col].dtypes=="object"]
    cat_cols=cat_cols+num_but_cat
    cat_cols=[col for col in cat_cols if col not in cat_but_car]
    num_cols=[col for col in dataframe.columns if dataframe[col].dtypes!="object"]
    num_cols=[col for col in num_cols if col not in num_but_cat]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    print(f"cat_but_car:{len(cat_but_car)}")
    # cat_cols + num_cols+ cat_but_car =değişken sayısı
    # num_but_cat cat_cols un içinde zaten, sadece raporlama için verilmiş
    return cat_cols,cat_but_car,num_cols,num_but_cat
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)
#6 num değişken var sadece, mümerik analize bakalım.

# NUMERICAL/analiz
def num_summary(dataframe,numerical_col,plot=False):
    quantiles=[0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99,1]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.ylabel(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

#Görev 2: K-Means ile Müşteri Segmentasyonu

#Adım 1: Değişkenleri standartlaştırınız

sc = MinMaxScaler((0,1)) #bütün değerler 0,1 aralığında düzenlendi.
df = sc.fit_transform(df)
df[0:5]

#Adım 2: Optimum küme sayısını belirleyiniz.

kmeans = KMeans()
ssd = []
K = range(1, 30)
for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)
#1 den 30 a kadar küme sayısına göre sse leri oluşturduk.sonra görselleştirdik.
plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()
#Elbow(dirsek) yöntemi bize küme sayısının kaç olması gerektiğiyle ilgili bilgi veriyor.
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_ #optimum küme sayısını 6 olarak verdi.

#Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)
clusters_kmeans = kmeans.labels_ #burada hangisi hangi merkeze bağlıysa onu etiketlendirdik.
df = pd.read_csv("datasets/flo_data_20k.csv")
df["kmeans_cluster"] = clusters_kmeans
df.head()

#cluster larına göre bi analiz yapmak istersek:

df.groupby("kmeans_cluster").agg(["count","mean","median"])

#Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu

#Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz

sc = MinMaxScaler((0,1)) #bütün değerler 0,1 aralığında düzenlendi.
df = sc.fit_transform(df)
hc_average = linkage(df, "average")

# Kume Sayısını Belirlemek
plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y=0.5, color='r', linestyle='--')#0.5noktasından çizgi atıp kümelemesini istedik.
plt.axhline(y=0.6, color='b', linestyle='--')#0.6 noktasından çizgi atıp kümeleme istedik.
plt.show()

#final modeli oluştur.
cluster = AgglomerativeClustering(n_clusters=5, linkage="average")
clusters = cluster.fit_predict(df)
df = pd.read_csv("datasets/flo_data_20k.csv")
df["hi_cluster_no"] = clusters
df["kmeans_cluster_no"] = clusters_kmeans
df.head()

#analiz
df.groupby("hi_cluster_no").agg(["count","mean","median"])






