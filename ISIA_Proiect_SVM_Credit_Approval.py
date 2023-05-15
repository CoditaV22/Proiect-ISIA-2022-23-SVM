# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:35:10 2022
Proiect ISIA SVM 2022
Nume:Codita Vlad-Alexandru
Grupa:424A
"""

"Bibliotecile utilizate:"

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

"Vector in care am adaugat toate variatiile functiei cost:"

c = [0.00001,0.03125 , 0.125 , 0.5 , 2 , 8 , 32 , 128,0.084]

"Importarea datelor: "

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
df = pd.read_csv(url , 
                 names=['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12','A13','A14','A15','A16']) 
  
"Missing Data Problem:"
"Am ales sa completez liniile in care aveam date lipsa deoarece reprezentau un procent relevant in comparatie cu restul de date(37 linii din 690)(5%)"
df=df.replace('?', np.NaN)
df['A2'] = df['A2'].fillna(df['A2'].median())
df['A14'] = df['A14'].fillna(df['A14'].median())
df['A1'] = df['A1'].fillna(df['A1'].mode()[0])
df['A7'] = df['A7'].fillna(df['A7'].mode()[0])
df['A6'] = df['A6'].fillna(df['A6'].mode()[0])
df['A5'] = df['A5'].fillna(df['A5'].mode()[0])
df['A4'] = df['A4'].fillna(df['A4'].mode()[0])


X=df.drop('A16',axis=1).copy()
Y=df['A16'].copy()


X=pd.get_dummies(X,columns=['A1', 'A4' , 'A5' , 'A6' , 'A7' , 'A9' , 'A10' , 'A12' , 'A13' ])

scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
X = scaling.transform(X)

"Impartirea datelor in train/test(75/25)"
X_test,X_train,Y_test,Y_train= train_test_split(X,Y,train_size= 0.25,random_state=0)

for i,cost in enumerate(c):
    clf=svm.SVC(kernel='linear',C=cost,gamma = 1)
    clf.fit(X_train,Y_train)
    pred = clf.predict(X_test)
    print('Pentru cost= ' + str(cost) +  ', acuratetea este ' + str(accuracy_score(Y_test,pred) * 100) + '%')
    
    
