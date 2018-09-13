from Network import Network
import numpy as np
import mnist

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

net = Network((784,100,10),1)
net.fit(x,y,epochs=40)

i=0
for a in x:
    print(lbl[i])
    o = net.forward(a)
    print(o.argmax())
    i+=1
    input()
