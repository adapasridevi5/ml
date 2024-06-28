import pandas as pd

data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\CarPrice_Assignment.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

cols = ['CarName','carbody','drivewheel','enginetype','fuelsystem']

for i in cols:
    data[i] = le.fit_transform(data[i])

x = data.iloc[ : , :-1].values
y = data.iloc[ : ,-1].values

from sklearn.model_selection import train_test_split

xtrain , xtest , ytrain , ytest = train_test_split(x,y,test_size=0.3,random_state=45)

from sklearn.tree import DecisionTreeRegressor
dl = DecisionTreeRegressor(random_state=45)

dl.fit(xtrain,ytrain)

ypred = dl.predict(xtest)

from sklearn.metrics import r2_score

rscore = r2_score(ytest,ypred)

print(rscore*100)