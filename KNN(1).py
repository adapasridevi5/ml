import pandas as pd

data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\dataset.csv")

x = data.iloc[ 0:200, 1:-6].values
y = data.iloc[ 0:200,-1].values

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=80)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7)

model.fit(xtrain,ytrain)

y_pred = model.predict(xtest)

print("ytest values are ")
print(ytest)

print("ytest values predicted by model are")
print(y_pred)


from sklearn.metrics import confusion_matrix,accuracy_score

accuracy = accuracy_score(y_pred,ytest)
cmatrix = confusion_matrix(y_pred,ytest)

print("Accuracy - ",accuracy*100)
print("Confusion Matrix - ",cmatrix)