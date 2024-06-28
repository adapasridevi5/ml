import pandas as pd
 
data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\iris.csv")

data = data.drop(['color'],axis=1)

data['pw'] = data['pw'].fillna(data['pw'].mean())

x = data.iloc[ : , :-1].values
y = data.iloc[ : ,-1].values

from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(x,y,test_size=0.3,random_state= 42)

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=5)

model.fit(xtrain,ytrain)

ypred = model.predict(xtest)

from sklearn.metrics import accuracy_score

accs = accuracy_score(ytest,ypred)

print(accs*100)

