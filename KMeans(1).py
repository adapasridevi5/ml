import pandas as pd

data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\iris.csv")

data['pw'] = data['pw'].fillna(data['pw'].mean())

data = data.drop('class',axis=1)

x = data.iloc[ : , [0,2]].values

from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)

ypred = model.fit_predict(x)

print(ypred)

