import pandas as pd

data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\diabetes.csv")

x = data.iloc[ : , :-1].values
y = data.iloc[ : , -1].values

# x = data.drop(["outcome"], axis=1)
# y = data["outcome"]

from sklearn.model_selection import train_test_split

xtrain , xtest , ytrain , ytest = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)

model.fit(xtrain,ytrain)

ypred = model.predict(xtest)

from sklearn.metrics import accuracy_score

acc = accuracy_score(ypred,ytest)

print(acc*100)
