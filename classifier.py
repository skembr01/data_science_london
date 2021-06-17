# from numpy.core.fromnumeric import ravel
from numpy.core.fromnumeric import shape
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import csv
# from tensorflow.keras import layers
# from tensorflow.keras import Sequential
# from tensorflow.keras.optimizers import SGD
# from tensorflow.python.keras.engine.input_layer import InputLayer

#loading dataframes
train = pd.read_csv('train.csv', header=None)
train_labels = pd.read_csv('trainLabels.csv', header=None)
test = pd.read_csv('test.csv', header=None)



#train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(train, train_labels, test_size=0.2, random_state=2)
feature_shape = features_train.shape



#scale data
scaler = StandardScaler()
scaler.fit_transform(features_train)
scaler.fit(features_test)

#neural network
#set up model
# model = Sequential()
# model.add(InputLayer(input_shape=(features_train.shape[1],)))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# #compile model
# opt = SGD(learning_rate=0.01)
# model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# #fit model
# model.fit(features_train, labels_train, epochs=60, batch_size=800, verbose=0)

# #results
# ent, acc = model.evaluate(features_test, labels_test)
# print(ent,acc)
#logistic regression model = .50 score on kaggle
# regression = LogisticRegression()
# regression.fit(features_train, labels_train)

# #scoring model
# score = regression.score(features_test, labels_test)
# print(score)

#random forest model
model = RandomForestClassifier()
model.fit(features_train, labels_train.values.ravel())

#scoring model
score = model.score(features_test, labels_test)
print(score)

#getting prediction and making csv file
prediction = model.predict(test)
print(len(prediction))
prediction = np.around(prediction)
prediction = prediction.astype(int)



csv_id_rows = []
for i in range(1, 90001):
    csv_id_rows.append(i)
# print(len(csv_id_rows))
csv_val_rows = []
for value in prediction:
    csv_val_rows.append(value)
# print(len(csv_val_rows))
with open('london_sklearn.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Id', 'Solution'])
    for i in range(9000):
        writer.writerow([csv_id_rows[i], csv_val_rows[i]])
