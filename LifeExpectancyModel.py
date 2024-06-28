import pandas as pd
 
data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\LifeExpectancy.csv")

ncol = ['Gender','Prenatal Status','Marital Status','Education','Migrant Status']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in ncol:
    data[i] = le.fit_transform(data[i])

col=['Gender','Age','Heart Rate','BMI','Cholesterol','Life Expectancy']
for i in col:
    data[i] = data[i].fillna(data[i].mean())

x = data.iloc[ : , :-1].values
y = data.iloc[ : ,-1].values
  
from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(x,y,test_size=0.3,random_state=45)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=75)

model.fit(xtrain,ytrain)

ypred = model.predict(xtest)

from sklearn.metrics import r2_score

print(r2_score(ytest,ypred))
