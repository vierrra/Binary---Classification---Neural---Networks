import pandas as pd
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

csv = pd.read_csv('data.csv', sep=',')
csv['fruta'] = le.fit_transform(csv['fruta'])

data = csv.values

attribute   = data[:, 2:]
classifiers = data[:, 1]

model = Sequential()
model.add(Dense(units=2, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(attribute, classifiers, batch_size=30, epochs=100)

response = model.predict(np.array([
    [3.1, 122],
    [4.1, 146],
    [2.2, 86]
]))

for result in response:
    print("%.6f" % result[0])
    if result[0] < 0.5:
        print("Laranja")
    else:
        print("Limão")