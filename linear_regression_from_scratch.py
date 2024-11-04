import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import warnings
sns.set_theme(style='darkgrid')
warnings.simplefilter("ignore")

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

#data regularization
X['house_age'] = (X['house_age'] - np.mean(X['house_age'], axis=0)) / np.std(X['house_age'], axis=0)
X['distance_to_the_nearest_MRT_station'] = (X['distance_to_the_nearest_MRT_station'] - np.mean(X['distance_to_the_nearest_MRT_station'], axis=0)) / np.std(X['distance_to_the_nearest_MRT_station'], axis=0)
X['number_of_convenience_stores'] = (X['number_of_convenience_stores'] - np.mean(X['number_of_convenience_stores'], axis=0)) / np.std(X['number_of_convenience_stores'], axis=0)
X['latitude'] = (X['latitude'] - np.mean(X['latitude'], axis=0)) / np.std(X['latitude'], axis=0)
X['longitude'] = (X['longitude'] - np.mean(X['longitude'], axis=0)) / np.std(X['longitude'], axis=0)

#parameters initialization
w = [1 , 1 , 1 , 1 , 1 , 1, 1]
b = 0
n_iterations = 1000
learning_rate = 0.1

#split data to train and test  
X_train = X.values[:math.floor((data.shape[0]) * 0.8) , :]
X_test = X.values[len(X_train): , :]
y_train = y[:len(X_train)]
y_test = y[len(X_train):]

#lossfunc
def MSE():
    N = len(X_train)
    y_pred = X_train.dot(w) + b
    mse = (1 / N) * np.sum((y_pred - y_train) ** 2) 
    return mse

#gradient
def grad_func():
    N = len(X_train)
    y_pred = X_train.dot(w) + b
    error = y_pred - y_train
    
    dw = (2 / N) * X_train.T.dot(error)
    db = (2 / N) * np.sum(error)
    
    return dw, db

#start of learning
for i in range(1 ,n_iterations +1):
    dw, db = grad_func()
    
    # updating the weights
    w -= learning_rate * dw
    b -= learning_rate * db
    mse = MSE()
    if i % 100 == 0:
        print(f"Iteration {i}: MSE = {mse}")


#predict the price
y_predict = []
for i, value in enumerate(X_test[:]):
    # Прогнозирование значений
    predicted_value = (value[0]*w[0] + value[1]*w[1] + value[2]*w[2] + value[3]*w[3] + value[4]*w[4] + value[5]*w[5] + value[6]*w[6] + b)
    # Добавляем разницу между прогнозом и реальным значением
    y_predict.append(predicted_value)

y_predict = np.array(y_predict)
y_predict = y_predict.tolist()

#testing the predict
def test():
    print(r2_score(y_test, y_predict , multioutput='uniform_average'))
    print(mean_absolute_error(y_test, y_predict))
    print(mean_absolute_percentage_error(y_test, y_predict))

test()

#final result
plt.scatter(y_test, y_test, color="black")
plt.plot(y_test, y_predict, color="blue", linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
