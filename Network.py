#-----------------------------------------------#
# autor: Leonardo de Abreu Schmidt              #
# Network: artificial neural network class      #
#-----------------------------------------------#

import numpy as np
import math
import time

class Network():

    def __init__(self,architecture,tr):
        self.weights = []
        self.delta = []
        self.architecture = architecture
        self.layers = []
        self.bias = []
        self.dbias = []
        self.tr = tr
        print(self.architecture[-1:])
        self.fme = (1.0/self.architecture[-1:][0])
        for w in range(len(architecture)-1):
            x = architecture[w]
            y = architecture[w+1]
            
            wts = np.random.uniform(-1.0,1.0,(y,x))
            self.weights.append(wts)
            dwts = np.random.uniform(-1.0,1.0,(y,x))
            self.delta.append(dwts)

            bs = np.random.uniform(-1.0,1.0,(y))
            self.bias.append(bs)
            dbs = np.random.uniform(-1.0,1.0,(y))
            self.dbias.append(dbs)

            self.layers.append( np.array([[0.0]*architecture[w+1]]*2 ))

    def fit(self,X,Y,epochs=10):
        for ep in range(epochs):
            err=0.0
            i=0
            hits=0
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

                #--------------------------error----------------------------------
                if(np.argmax(out_layer)==np.argmax(b)):
                    hits+=1
                error = b-out_layer
                err += (self.fme*sum(error*error))
                #--------------------------error----------------------------------

                rlayers = list(reversed(self.layers))
                rweights = list(reversed(self.weights))
                nlayers = len(self.layers)

                #---------------------backpropagation-----------------------------
                l=0
                ns = len(rlayers[l][0])
                
                dv = self.dsigmoid(rlayers[l][0])
                grad = -error*dv
                self.delta[nlayers-(l+1)] = -self.tr*self.mult(grad,rlayers[l+1][1])
                self.dbias[nlayers-(l+1)] = -self.tr*grad*[1.0]*ns

                l+=1
                while(l<nlayers):
                    
                    ns = len(rlayers[l][0])

                    dv = self.dsigmoid(rlayers[l][0])
                    gr = np.array([grad])
                    grad = dv*sum(gr.T*rweights[l-1])
                    if(l<nlayers-1):
                        out_layer = rlayers[l+1][1]
                    else:
                        out_layer = a
                    self.delta[nlayers-(l+1)] = -self.tr*self.mult(grad,out_layer)
                    self.dbias[nlayers-(l+1)] = -self.tr*grad*[1.0]*ns
                    l+=1

                for x in range(nlayers):
                    self.weights[x] += self.delta[x]
                    self.bias[x]    += self.dbias[x]
                #---------------------backpropagation-----------------------------
                i+=1
            print("epoch "+str(ep)+", error: "+str(err/len(X))+", accuracy: "+str(hits/len(X)))

    def mult(self,v,u):
        h=[u]*len(v)
        for x in range(len(v)):
            h[x]=v[x]*u
        h = np.array(h)
        return h

    def forward(self,a):
        w=0
        for l in self.layers:
            l[0] = sum((a*self.weights[w]).T)+self.bias[w]
            a=self.sigmoid(l[0])
            w+=1
        return a

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