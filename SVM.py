
# Çalışmanın detaylı anlatımına https://burakzdd.medium.com/support-vector-machine-destek-vekt%C3%B6r-makinesi-svm-machine-learning-54fbdd47029d linkten ulaşabilirsiniz
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

veri=pd.read_csv("  ") #parantez içine veri setinin dosya yolu girilmelidir

X1=veri[‘X1’]
X2=veri[‘X2’]
X_training=np.array(list(zip(X1,X2)))
X_training
y_training=veri[‘y’]
y_training
target_names=[‘-1’,’+1']
target_names
              
idxPlus=y_training[y_training<0].index
idxMin=y_training[y_training>0].index
plt.scatter(X_training[idxPlus,0],X_training[idxPlus,1],c=’purple’,s=50)
plt.scatter(X_training[idxMin,0],X_training[idxMin,1],c=’pink’,s=50)
plt.legend(target_names,loc=2)
plt.xlabel(‘X1’)
plt.ylabel(‘X2’);
plt.title(‘SVM Sııflandırma’)
plt.show()
              
svc = svm.SVC(kernel=’linear’).fit(X_training,y_training)
svc.get_params(True)   
              
lbX1=math.floor(min(X_training[:,0]))-1
ubX1=math.ceil(max(X_training[:,0]))+1
lbX2=math.floor(min(X_training[:,1]))-1
ubX2=math.ceil(max(X_training[:,1]))+1
[lbX1,ubX1,lbX2,ubX2]
              
idxPlus=y_training[y_training<0].index
idxMin=y_training[y_training>0].index
plt.scatter(X_training[idxPlus,0],X_training[idxPlus,1],c=’blue’,s=50)
plt.scatter(X_training[idxMin,0],X_training[idxMin,1],c=’green’,s=50)
plt.legend(target_names,loc=2)
X,Y = np.mgrid[lbX1:ubX1:100j,lbX2:ubX2:100j]
Z = svc.decision_function(np.c_[X.ravel(),Y.ravel()])
Z = Z.reshape(X.shape)
plt.contourf(X,Y,Z > 0,alpha=0.4)
plt.contour(X,Y,Z,colors=[‘red’], linestyles=[‘ — ‘],levels=[0])
plt.title(‘Doğrusal olarak ayrılmış’)
plt.show()
              
idxPlus=y_training[y_training<0].index
idxMin=y_training[y_training>0].index
plt.scatter(X_training[idxPlus,0],X_training[idxPlus,1],c=’b’,s=50)
plt.scatter(X_training[idxMin,0],X_training[idxMin,1],c=’g’,s=50)
plt.legend(target_names,loc=2)
X,Y = np.mgrid[lbX1:ubX1:100j,lbX2:ubX2:100j]
Z = svc.decision_function(np.c_[X.ravel(),Y.ravel()])
Z = Z.reshape(X.shape)
plt.contourf(X,Y,Z > 0,alpha=0.4)
plt.contour(X,Y,Z,colors=[‘r’,’k’,’r’], linestyles=[‘ — ‘,’-’,’ — ‘],levels=[-1,0,1])
plt.scatter(svc.support_vectors_[:,0],svc.support_vectors_[:,1],s=120,facecolors=’none’)
plt.scatter(X_training[:,0],X_training[:,1],c=y_training,s=50,alpha=0.95);
plt.title(‘Margin ve Destek vektörü’)
plt.show()

              
              
              
              
