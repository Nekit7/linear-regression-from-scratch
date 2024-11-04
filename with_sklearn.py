import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import sklearn.linear_model as linear_model
sns.set_theme(style='darkgrid')

# 1. Loading data from CSV
data = pd.read_csv('data.csv' , index_col='N')
#To view the data, uncomment the string below 
#print(data.head())

#to view the data correlation uncomment the strings below 
'''
plt.figure(figsize = (12,8))

ax = sns.heatmap(X.corr(), annot = True, fmt = ".2f")

i, k = ax.get_ylim()

ax.set_ylim(i+0.5, k-0.5)

plt.show()
'''

#Dropping useless data and creating a new features
data = data.drop(['transaction_date'] , axis=1)
data = data[data['distance_to_the_nearest_MRT_station'] <= 2800]
y = data["house_price_of_unit_area"]
X = data.drop(columns=["house_price_of_unit_area"])
def new_params():
    x6 = []
    x7 = []
    for i in range(data.shape[0]):
        x6.append(X.values[i][1]*y.values[i])
        x7.append(X.values[i][3]*y.values[i])
    
    return [(x6 - np.mean(x6, axis=0)) / np.std(x6, axis=0) ,(x7 - np.mean(x7, axis=0)) / np.std(x7, axis=0)]
X['x6'] , X['x7'] = new_params()

#To view the data visualisation, uncomment the strings below 
"""
fig, axs = plt.subplots(figsize=(16, 5), ncols=6)

for i, feature in enumerate(["distance_to_the_nearest_MRT_station", "number_of_convenience_stores", "latitude" , "longitude" , "x6" , "x7"]):

    axs[i].scatter(y_train,X_train[feature], alpha=0.6)

    axs[i].set_xlabel('house_price_of_unit_area')

    axs[i].set_ylabel(feature)

plt.tight_layout()

plt.show()
"""

#split data to train and test  
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42 ,test_size = 0.2)  

#craeting a model
model = linear_model.LinearRegression()

#model training
model.fit(X_train, y_train)

#predict the price
y_pred = model.predict(X_test) 

#metrics
print(r2_score(y_test, y_pred , multioutput='uniform_average'))
print(mean_absolute_error(y_test, y_pred))
print(mean_absolute_percentage_error(y_test, y_pred))

#final result
plt.scatter(y_test, y_test, color="black")
plt.plot(y_test, y_pred, color="blue", linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
