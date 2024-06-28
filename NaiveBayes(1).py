import pandas as pd

data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\dataset.csv")

x = data.iloc[ : , :-6].values
y = data.iloc[ : ,-1].values

from sklearn.model_selection import train_test_split
 
xtrain , xtest , ytrain , ytest = train_test_split(x,y,test_size = 0.3 , random_state = 42)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

model = model.fit(xtrain,ytrain)

ypred = model.predict(xtest)

from sklearn.metrics import accuracy_score,confusion_matrix

acc = accuracy_score(ytest,ypred)
cmatrix = confusion_matrix(ytest,ypred)

print(acc*100)
print(cmatrix)
            