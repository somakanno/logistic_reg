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


#偶数行目を訓練データとする
y_train = np.array([y[2*i] for i in range(50)],dtype = np.float64)
X_train = np.array([X[2*i] for i in range(50)],dtype = np.float64) 

#奇数行目を試験データとする
y_test = np.array([y[2*i+1] for i in range(50)],dtype = np.float64)
X_test = np.array([X[2*i+1] for i in range(50)],dtype = np.float64) 

#訓練データで学習
model = logisticreg.LogisticRegression(tol = 0.01)
model.fit(X_train,y_train)

#学習モデルを使って予測
y_predict = model.predict(X_test)
n_hits  = (y_test == y_predict).sum()
print('Accuracy:{}/{} = {}'.format(n_hits,n_test,
                                   n_hits/n_test))

