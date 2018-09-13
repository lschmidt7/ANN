class Metrics():

    def __init__(self):
        pass

    def accuraccy(self,y_true,y_pred):
        hit=0
        for x in range(len(y_pred)):
            if(y_pred[x] == y_true[x]):
                hit+=1
        return float(hit)/len(y_true)