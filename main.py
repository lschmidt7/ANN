from Network import Network
import numpy as np

x = np.array([[1.0,1.0],[0.0,0.0],[1.0,0.0],[0.0,1.0]])
y = np.array([1,1,0,0])

#x = np.random.uniform(0.0,1.0,(100,10))
#y = np.random.uniform(0.0,1.0,(100,2))

weights = np.array([[[0.8,0.3],[-0.6,-0.4]],[[0.7,-0.8]]])
bias = np.array([ [0.7,-0.4],[-0.3] ])

#net = Network((10,5,2))
net = Network((2,2,1),0.8)
net.set_weights(weights)
net.set_bias(bias)
net.fit(x,y,epochs=10000)

net.forward([0.1,0.1])
net.forward([0.9,0.9])
net.forward([0.1,0.9])
net.forward([0.7,0.1])