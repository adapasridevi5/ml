import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\Purchasedataset.csv")

x = data.iloc[ 1:10 , :-1]
y = data.iloc[ 1:10 ,-1]

from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(x,y,test_size=0.3,random_state=10)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)

model.fit(xtrain,ytrain)

y_pred = model.predict(xtest)


print(ytest)
print(y_pred)

from sklearn.metrics import accuracy_score,confusion_matrix

accuracy = accuracy_score(y_pred,ytest)
cmatrix = confusion_matrix(y_pred,ytest)
print(accuracy*100)
print(cmatrix)