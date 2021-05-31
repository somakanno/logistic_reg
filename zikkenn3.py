import logisticreg
import csv
import numpy as np
import random


n_test = 50
X = []
y = []
with open ('iris.data') as fp:
    for row in csv.reader(fp):
        if row[4] == "Iris-setosa":
            y.insert(random.randint(0,len(y)),0)
        else:
            y.insert(random.randint(0,len(y)),1)
        X.insert(random.randint(0,len(X)),row[:4])
        
y = np.array(y,dtype = np.float64)
X = np.array(X,dtype = np.float64)


print(len(y))
print(len(X))