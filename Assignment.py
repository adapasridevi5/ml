import pandas as pd

data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\medical_students_dataset.csv")

col = ['Student ID','BMI','Smoking']
for i in col:
    data = data.drop(i,axis=1)

pcol = ['Age','Height','Weight','Temperature','Heart Rate','Blood Pressure','Cholesterol']
for i in pcol:
    data[i] = data[i].fillna(data[i].mean())

scol = ['Gender','Blood Type','Diabetes']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for i in scol:
    data[i] = le.fit_transform(data[i])
    data[i] = data[i].fillna(data[i].mean())

x = data.iloc[ : , :-1].values
y = data.iloc[ : ,-1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state= 900)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=700)

model.fit(xtrain,ytrain)

ypred = model.predict(xtest)

from sklearn.metrics import accuracy_score
acc = accuracy_score(ytest,ypred)

print(acc*100)