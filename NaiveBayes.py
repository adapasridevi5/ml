import pandas as pd

data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\iris.csv")

#discarding color column
data = data.drop(['color'],axis = 1)

#discarding null value columns only if they are less than 40% in the column
#data = data.dropna()

#Filling out the null value with mean of the column
data['pw'] = data['pw'].fillna(data['pw'].mean())

x = data.iloc[ : , : -1].values
y = data.iloc[ : , -1].values

from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain, ytest = train_test_split(x,y,test_size=0.25,random_state=25)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

model.fit(xtrain , ytrain)

ypred = model.predict(xtest)

from sklearn.metrics import accuracy_score,confusion_matrix

acc = accuracy_score(ytest,ypred)
cmatrix = confusion_matrix(ytest,ypred)

print(acc*100)
print(cmatrix)

