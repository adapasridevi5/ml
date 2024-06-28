from sklearn.datasets import load_iris

data=load_iris()
x = data.data
y = data.target

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.25 , random_state= 45)

from sklearn.neighbors import KNeighborsClassifier
model= KNeighborsClassifier(n_neighbors=5)
model.fit(xtrain,ytrain)
ypred = model.predict(xtest)

from sklearn.metrics import accuracy_score,confusion_matrix
accuracy = accuracy_score(ypred,ytest)
cmatrix = confusion_matrix(ypred,ytest)

print(accuracy*100)
print(cmatrix)


