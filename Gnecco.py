#!/usr/bin/env python
# coding: utf-8

 #In[3]:


import matplotlib.pyplot as plt
#matplotlib inline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv("entrenamiento.csv")
num = np.array(df)
#df['Registro']= pd.to_datetime(df['Registro'], format='%m/%d/%Y %H:%M')

from math import sin, cos, sqrt, atan2, radians

def getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2):
    R = 6371 # Radius of the earth in km
    dLat = radians(lat2-lat1)
    dLon = radians(lon2-lon1)
    rLat1 = radians(lat1)
    rLat2 = radians(lat2)
    a = sin(dLat/2) * sin(dLat/2) + cos(rLat1) * cos(rLat2) * sin(dLon/2) * sin(dLon/2) 
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    d = R * c # Distance in km
    return d

def calc_velocity(dist_km, time_start, time_end):
    """Return 0 if time_start == time_end, avoid dividing by 0"""
    return dist_km / (time_end - time_start).seconds if time_end > time_start else 0

pro_id = num[:,1]
Registo = num[:,3]
Latitud = num[:,5]
Longitud = num[:,7]
Frecuencia = num[:,9]
PrecionBarometric = num[:,11]

Diccionario = {
    'id' : pro_id,
    'Registro' : Registo,
    'Latitud' : Latitud,
    'Longitud' : Longitud,
    'Frecuencia' : Frecuencia,
    'PrecionBarometrica' : PrecionBarometric
};

df = pd.DataFrame(Diccionario)

df['Registro'] = pd.to_datetime(df['Registro'], format='%d/%m/%y %H:%M:%S')



df =  df.sort_values(by=['id', 'Registro'])

df['lat0'] = df.groupby('id')['Latitud'].transform(lambda x: x.iat[0])
df['lon0'] = df.groupby('id')['Longitud'].transform(lambda x: x.iat[0])
df['t0'] = df.groupby('id')['Registro'].transform(lambda x: x.iat[0])
df['Distancia'] = df.apply(
    lambda row: getDistanceFromLatLonInKm(
        lat1=row['Latitud'],
        lon1=row['Longitud'],
        lat2=row['lat0'],
        lon2=row['lon0']
    ),
    axis=1
)

df['Velocidad'] = df.apply(
    lambda row: calc_velocity(
        dist_km=row['Distancia'],
        time_start=row['t0'],
        time_end=row['Registro']
    ),
    axis=1
)
print(df)
filter_df = df[(df['Frecuencia'] <= 3500) & (df["Velocidad"] <= 80000)]
data_x = filter_df[['Frecuencia']]
x_train = np.array(data_x)
y_train = filter_df['Velocidad'].values
print(x_train)

regr = linear_model.LinearRegression()
regr.fit(x_train,y_train)
y_pre = regr.predict(x_train)
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,y_pre,color='blue',linewidth=3)

plt.show()


