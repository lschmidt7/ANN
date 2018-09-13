import numpy as np
import math

class Network():

    def __init__(self,architecture,tr):
        self.weights = []
        self.architecture = architecture
        self.layers = []
        self.bias = []
        self.tr = tr
        for w in range(len(architecture)-1):
            x = architecture[w]
            y = architecture[w+1]
            wts = np.random.uniform(0.0,1.0,(y,x))
            self.weights.append(wts)
            bs = np.random.uniform(0.0,1.0,(y))
            self.bias.append(bs)
            self.layers.append( np.array([[0.0]*architecture[w+1]]*3 ))

    def soma(self,v):
        v1 = []
        for x in v:
            sm=0
            for y in x:
                sm+=y
            v1.append(sm)
        return v1

    def fit(self,X,Y,epochs=10):
        for ep in range(epochs):
            err=0.0
            i=0
            for a in X:
                b = Y[i]
                w=0
                out_layer = a
                #---------------------forward-------------------------------------
                for l in self.layers:
                    l[0] = sum((out_layer*self.weights[w]).T)+self.bias[w]
                    out_layer=self.sigmoid(l[0])
                    l[1] = out_layer
                    w+=1
                #---------------------forward-------------------------------------
                error = b-out_layer
                err += (0.1*sum(error*error))
                rlayers = list(reversed(self.layers))
                rweights = list(reversed(self.weights))
                #---------------------backpropagation-----------------------------
                l=0
                ns = len(rlayers[l][0]) # neurons in the current layer
                
                dv = self.dsigmoid(rlayers[l][0])
                grad1 = -error*dv
                delta_weights1 = -self.tr*self.mult(grad1,rlayers[l+1][1])
                delta_bias1 = -self.tr*grad1*[1.0]*ns # cuidar aqui

                self.weights[1] += delta_weights1
                self.bias[1]    += delta_bias1

                l+=1
                ns = len(rlayers[l][0]) # neurons in the current layer

                dv2 = self.dsigmoid(rlayers[l][0])
                gr = np.array([grad1])
                grad2 = dv2*sum(gr.T*rweights[l-1])
                delta_weights2 = -self.tr*self.mult(grad2,a)
                delta_bias2 = -self.tr*grad2*[1.0]*ns # cuidar aqui

                self.weights[0] += delta_weights2
                self.bias[0]    += delta_bias2
                #---------------------backpropagation-----------------------------
                i+=1
            print("epoch "+str(ep)+", error: "+str(err/len(X)))

    def mult(self,v,u):
        h=[]
        for x in v:
            h.append(x*u)
        h = np.array(h)
        return h

    def forward(self,a):
        out_layer=a
        w=0
        for l in self.layers:
            l[0] = sum((out_layer*self.weights[w]).T)+self.bias[w]
            v=l[0]
            out_layer=self.sigmoid(v)
            l[1] = out_layer
            w+=1
        print(out_layer)

    def show_weights(self):
        for l in range(len(self.architecture)-1):
            print(self.weights[l].shape)
    
    def show_layers(self):
        for l in self.layers:
            print(l)
    
    def show_bias(self):
        for b in self.bias:
            print(b.shape)
    
    def sigmoid(self,v):
        return 1.0/(1.0+pow(math.e,-1.0*v))
    
    def dsigmoid(self,v):
        return pow(math.e,-1.0*v)/pow((1.0+pow(math.e,-1.0*v)),2)
    
    def set_weights(self,weights):
        self.weights = weights
    
    def set_bias(self,bias):
        self.bias = bias