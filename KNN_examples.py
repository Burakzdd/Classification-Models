"""
Created on Wed Mar  3 08:37:51 2021
@author: burakzdd
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# 25 bilinen / eğitim verisinin (x, y) değerlerini içeren özellik seti
trainData = np.random.randint(0,100,(25,2)).astype(np.float32)
# Her bir yeşil ve sarı daireyi 0 ve 1 rakamlarıyla etiketleyin
responses = np.random.randint(0,2,(25,1)).astype(np.float32)
# Yeşil komşu noktaları al ve grafikte konumlandır
yesil = trainData[responses.ravel()==0]
plt.scatter(yesil[:,0],yesil[:,1],80,'g','o')
# Sarı komşu noktaları al ve grafikte konumlandır
sarı = trainData[responses.ravel()==1]
plt.scatter(sarı[:,0],sarı[:,1],80,'y','o')

#yeni mavi nokta eklenir
newcomer = np.random.randint(0,100,(1,2)).astype(np.float32)
plt.scatter(newcomer[:,0],newcomer[:,1],80,'b','o')
plt.show()
#KNN uygulanır
knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, results, neighbours ,dist = knn.findNearest(newcomer, 3)
#Yeni eklenen verinin hangi sınıfta yer aldığı, mesafeler ve komşu pikseller bastırılır.
print( "Sonuç:  {}\n".format(results) )
print( "Komşular:  {}\n".format(neighbours) )
print( "Mesafe:  {}\n".format(dist) )

#Çalışmanın detaylı anlatımına https://burakzdd.medium.com/knn-k-en-yak%C4%B1n-kom%C5%9Fuluk-alogritmas%C4%B1-k-nearest-neighbor-8de3914f913e bu linkten ulaşabilitsiniz.
