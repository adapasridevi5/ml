import pandas as pd

data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\walmart.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

cols = ['Stay_In_Current_City_Years','Product_ID','Gender','City_Category','Age']

for i in cols:
    data[i] = le.fit_transform(data[i])
    
x = data.iloc[ : , :-1].values
y = data.iloc[ : ,-1].values

from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(x,y,test_size=0.3,random_state=50)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(xtrain,ytrain)

ypred = lr.predict(xtest)

from sklearn.metrics import r2_score

score = r2_score(ytest,ypred)

print(score*100)