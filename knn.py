# -*- coding: utf-8 -*-
"""KNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tI_-PEowKDObXG58ekj7fXV7dIe9UcLs
"""

#kütüphaneler import edilir
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.neighbors import KNeighborsRegressor

from warnings import filterwarnings
filterwarnings('ignore')

#dosyadan veriler okunarak model oluşturulur
hit = pd.read_csv("/content/Hitters.csv")
df = hit.copy()
df = df.dropna()
dms = pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y = df["Salary"]
X_ = df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X = pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.25, 
                                                    random_state=42)

# model knn algoritmasına göre fit edilir.
knn_model = KNeighborsRegressor().fit(X_train, y_train)

knn_model

#modelin komşuluk sayısına bakalım
knn_model.n_neighbors

#test veri setine göre modelin tahminleri
knn_model.predict(X_test)

y_tahmin = knn_model.predict(X_test)

#yapılan tahmin değerlerine göre hata ortalamasına bakalım
np.sqrt(mean_squared_error(y_test, y_tahmin))

#farklı k değerleri için karelerinin ortalama hatasıne bakılır
RMSE = [] 

for k in range(10):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors = k).fit(X_train, y_train)
    y_pred = knn_model.predict(X_train) 
    rmse = np.sqrt(mean_squared_error(y_train,y_pred)) 
    RMSE.append(rmse) 
    print("k =" , k , "için RMSE değeri: ", rmse)

from sklearn.model_selection import GridSearchCV

#knn parametleri belilenir
knn_params = {"n_neighbors": np.arange(1,30,1)}

#knn ye görte model yeniden fit edilir
knn = KNeighborsRegressor()

knn_cv_model = GridSearchCV(knn, knn_params, cv=10)

knn_cv_model.fit(X_train, y_train)

knn_cv_model.best_params_["n_neighbors"]

# oluşturulan iki modele bakılır. Gözelem yapıldığında ikinci mdoel için farklı k değerlerine göre sonuçların yakın değerler olduğu görülmektedir
RMSE = [] 
RMSE_CV = []
for k in range(10):
    k = k+1
    knn_model = KNeighborsRegressor(n_neighbors = k).fit(X_train, y_train)
    y_pred = knn_model.predict(X_train) 
    rmse = np.sqrt(mean_squared_error(y_train,y_pred)) 
    rmse_cv = np.sqrt(-1*cross_val_score(knn_model, X_train, y_train, cv=10, 
                                         scoring = "neg_mean_squared_error").mean())
    RMSE.append(rmse) 
    RMSE_CV.append(rmse_cv)
    print("k =" , k , "için RMSE değeri: ", rmse, "RMSE_CV değeri: ", rmse_cv )

knn_tuned = KNeighborsRegressor(n_neighbors = knn_cv_model.best_params_["n_neighbors"])

knn_tuned.fit(X_train, y_train)

np.sqrt(mean_squared_error(y_test, knn_tuned.predict(X_test)))