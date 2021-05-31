import logisticreg
import csv
import numpy as np



n_test = 50
X = []
y = []
with open ('iris.data') as fp:
    for row in csv.reader(fp):
        if row[4] == "Iris-setosa":
            y.append(0)
        else:
            y.append(1)
        X.append(row[:4])

     
y = np.array(y,dtype = np.float64)
X = np.array(X,dtype = np.float64)

#100~150行目のデータを削除し、二値データに変換
y = np.delete(y,slice(100,150),0)
X = np.delete(X,slice(100,150),0)

print(np.block([y, X]))