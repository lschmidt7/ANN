from keras.models import Sequential
from keras.layers import Dense
import keras
import numpy as np
import mnist

inp = 784

img,lbl = mnist.read(dataset = "training", path = "")

img = img[:1000]
lbl = lbl[:1000]

x=[]
for u in img:
    v = []
    for l in u:
        v.extend(l/255.0)
    x.append(v)
x = np.array(x)

y = []
for a in lbl:
    v = [0.0]*10
    v[a] = 1.0
    y.append(v)
y = np.array(y)

model = Sequential()
model.add(Dense(100, input_dim=inp, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
model.fit(x,y, epochs=40, batch_size=1)
