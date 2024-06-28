import pandas as pd
import matplotlib.pyplot as mtp

data = pd.read_csv(r"C:\Users\adapa\OneDrive\Desktop\THUB\CSV Files\countrydata.csv")

x = data.iloc[ : , [6,9]].values

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)

#centers = model.cluster_centers_

y_predict = model.fit_predict(x)

print(y_predict)

mtp.scatter(x[y_predict == 0, 0], x[y_predict == 0, 1], s = 50, c = 'blue', label = 'Cluster 1') #for first cluster  
mtp.scatter(x[y_predict == 1, 0], x[y_predict == 1, 1], s = 50, c = 'green', label = 'Cluster 2') #for second cluster  
mtp.scatter(x[y_predict== 2, 0], x[y_predict == 2, 1], s = 50, c = 'red', label = 'Cluster 3') #for third cluster  
#mtp.scatter(x[y_predict == 3, 0], x[y_predict == 3, 1], s = 50, c = 'cyan', label = 'Cluster 4') #for fourth cluster  
#mtp.scatter(x[y_predict == 4, 0], x[y_predict == 4, 1], s = 50, c = 'magenta', label = 'Cluster 5') #for fifth cluster  
mtp.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 80, c = 'yellow', label = 'Centroid')   
mtp.title('Clusters of countries')  
mtp.xlabel('Inflation')  
mtp.ylabel('GDP')  
mtp.legend()  
mtp.show()  