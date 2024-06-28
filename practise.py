# #KNN
# import pandas as pd
# data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\iris.csv")
# data['pw'] = data['pw'].fillna(data['pw'].mean())
# x=data.iloc[ : , :-1].values
# y=data.iloc[ : ,-1].values
# from sklearn.model_selection import train_test_split
# xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=45)
# from sklearn.neighbors import KNeighborsClassifier
# model = KNeighborsClassifier(n_neighbors=3)
# model.fit(xtrain,ytrain)
# ypred = model.predict(xtest)
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(ytest,ypred)
# print(acc*100)

#NaiveBayes
# import pandas as pd
# data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\iris.csv")
# data['pw'] = data['pw'].fillna(data['pw'].mean())
# x=data.iloc[ : , :-1].values
# y=data.iloc[ : ,-1].values
# from sklearn.model_selection import train_test_split
# xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=45)
# from sklearn.naive_bayes import GaussianNB
# model = GaussianNB()
# model.fit(xtrain,ytrain)
# ypred = model.predict(xtest)
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(ytest,ypred)
# print(acc*100)

#DecisionTreeClassifier
# import pandas as pd
# data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\iris.csv")
# data['pw'] = data['pw'].fillna(data['pw'].mean())
# x=data.iloc[ : , :-1].values
# y=data.iloc[ : ,-1].values
# from sklearn.model_selection import train_test_split
# xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=45)
# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier(random_state=8)
# model.fit(xtrain,ytrain)
# ypred = model.predict(xtest)
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(ytest,ypred)
# print(acc*100)

#DecisionTreeRegressor
# import pandas as pd
# data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\carPrice_Assignment.csv")
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# cols = ['CarName','carbody','drivewheel','enginetype','fuelsystem']
# for i in cols:
#     data[i] = le.fit_transform(data[i])
# x=data.iloc[ : , :-1].values
# y=data.iloc[ : ,-1].values
# from sklearn.model_selection import train_test_split
# xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=45)
# from sklearn.tree import DecisionTreeRegressor
# model = DecisionTreeRegressor(random_state=87)
# model.fit(xtrain,ytrain)
# ypred = model.predict(xtest)
# from sklearn.metrics import r2_score
# acc = r2_score(ytest,ypred)
# print(acc)

#RandomForestClassifier
# import pandas as pd
# data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\iris.csv")
# data['pw'] = data['pw'].fillna(data['pw'].mean())
# x=data.iloc[ : , :-1].values
# y=data.iloc[ : ,-1].values
# from sklearn.model_selection import train_test_split
# xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=45)
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier(random_state=85)
# model.fit(xtrain,ytrain)
# ypred = model.predict(xtest)
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(ytest,ypred)
# print(acc*100)

#RandomForestRegressor
# import pandas as pd
# data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\carPrice_Assignment.csv")
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# cols = ['CarName','carbody','drivewheel','enginetype','fuelsystem']
# for i in cols:
#     data[i] = le.fit_transform(data[i])
# x=data.iloc[ : , :-1].values
# y=data.iloc[ : ,-1].values
# from sklearn.model_selection import train_test_split
# xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=45)
# from sklearn.ensemble import RandomForestRegressor
# model = RandomForestRegressor(random_state=87)
# model.fit(xtrain,ytrain)
# ypred = model.predict(xtest)
# from sklearn.metrics import r2_score
# acc = r2_score(ytest,ypred)
# print(acc)

#LinearRegression
# import pandas as pd
# data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\carPrice_Assignment.csv")
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# cols = ['CarName','carbody','drivewheel','enginetype','fuelsystem']
# for i in cols:
#     data[i] = le.fit_transform(data[i])
# x=data.iloc[ : , :-1].values
# y=data.iloc[ : ,-1].values
# from sklearn.model_selection import train_test_split
# xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=45)
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(xtrain,ytrain)
# ypred = model.predict(xtest)
# from sklearn.metrics import r2_score
# acc = r2_score(ytest,ypred)
# print(acc)

#LogisticRegression
# import pandas as pd
# data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\iris.csv")
# data['pw'] = data['pw'].fillna(data['pw'].mean())
# x=data.iloc[ : , :-1].values
# y=data.iloc[ : ,-1].values
# from sklearn.model_selection import train_test_split
# xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=45)
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(random_state=85)
# model.fit(xtrain,ytrain)
# ypred = model.predict(xtest)
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(ytest,ypred)
# print(acc*100)

#Support Vector Classifier
# import pandas as pd
# data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\iris.csv")
# data['pw'] = data['pw'].fillna(data['pw'].mean())
# x=data.iloc[ : , :-1].values
# y=data.iloc[ : ,-1].values
# from sklearn.model_selection import train_test_split
# xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=45)
# from sklearn.svm import SVC
# model = SVC()
# model.fit(xtrain,ytrain)
# ypred = model.predict(xtest)
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(ytest,ypred)
# print(acc*100)

#Support Vector Regression
# import pandas as pd
# data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\carPrice_Assignment.csv")
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# cols = ['CarName','carbody','drivewheel','enginetype','fuelsystem']
# for i in cols:
#     data[i] = le.fit_transform(data[i])
# x=data.iloc[ : , :-1].values
# y=data.iloc[ : ,-1].values
# from sklearn.model_selection import train_test_split
# xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=45)
# from sklearn.svm import SVR
# model = SVR()
# model.fit(xtrain,ytrain)
# ypred = model.predict(xtest)
# from sklearn.metrics import r2_score
# acc = r2_score(ytest,ypred)
# print(acc)

#KMeans
# import pandas as pd
# data=pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\iris.csv")
# data['pw']=data['pw'].fillna(data['pw'].mean(0))
# x = data.drop('class',axis=1)
# from sklearn.cluster import KMeans
# model = KMeans(n_clusters=5)
# ypred = model.fit_predict(x)
# print(ypred)