# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 09:39:02 2019

@author: Rabin maharjan
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle as p

iris_value= np.array([list(map(float,input().split()))])
url = pd.read_csv('iris.csv')
data = url.values
X = data[:,0:4]
y = url['species']
seed = 6
test_size = 0.33
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
model = LogisticRegression(solver='liblinear')
model.fit(X_train, Y_train)
filename = 'finalized_model.sav'
p.dump(model, open(filename, 'wb'))

result= model.predict(iris_value)
print(result)