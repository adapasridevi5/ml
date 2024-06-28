import pandas as pd

data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\iris.csv")

data['pw'] = data['pw'].fillna(data['pw'].mean())

x = data.iloc[ : , :-1].values
y = data.iloc[ : ,-1].values

from sklearn.model_selection import train_test_split

xtrain , xtest , ytrain , ytest = train_test_split(x,y,test_size=0.3,random_state=45)

from sklearn.linear_model import LogisticRegression
dl = LogisticRegression(random_state=43)

dl.fit(xtrain,ytrain)

ypred = dl.predict(xtest)

from sklearn.metrics import accuracy_score

rscore = accuracy_score(ytest,ypred)

print(rscore*100)