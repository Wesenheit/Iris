"""
List of classes used to transform constrained distributions to unconstrined space if you want
or automaticly limit them in constrined space without manual need to set -inf propabilities in function

"""
import numpy as np
from scipy.special import expit,logit,log_expit

class Sigmoid():
    def __init__(self,low,high):
        self.low=low
        self.high=high

    def __call__(self, x):
        if x>0:
            return expit(x)*(self.high-self.low)+self.low
        else:
            return np.exp(x)/(1+np.exp(x))*(self.high-self.low)+self.low
    def inverse(self,x):
        x01=(x-self.low)/(self.high-self.low)
        return logit(x01)

    def jacobian_term(self,x):
        return log_expit(x)+log_expit(-x)+np.log(self.high-self.low)

class Identity():

    def __init__(self,low=-np.inf,high=np.inf):
        self.low=low
        self.high=high

    def __call__(self,x):
        return x
    
    def inverse(self,x):
        return x
    
    def jacobian_term(self,x):
        if x>self.high or x<self.low:
            return -np.inf
        else:
            return 0

class Exp():

    def __init__(self):
        self.low=0
        self.high=np.inf

    def __call__(self,x):
        return np.exp(x)
    
    def inverse(self,x):
        return np.log(x)
    
    def jacobian_term(self,x):
        return x

def biject(args):
    def wrapper(func):
        def function(old_arg,**kwargs):
            new_args=np.zeros(old_arg.shape)
            for i in range(len(old_arg)):
                new_args[i]=args[i](old_arg[i])
            odp=0
            for i in range(len(old_arg)):
                odp=odp+args[i].jacobian_term(old_arg[i])
            if np.isneginf(odp):
                return -np.inf
            else:
                return odp+func(new_args,**kwargs)
        return function
    return wrapper

def transform(state,lista):
    temp=np.zeros(state.shape)
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if state[i,j]>lista[j].high:
                tes=lista[j].high-np.random.rand()*0.001
                print("start above upper bound, rounding down to: ",tes)
                temp[i,j]=lista[j].inverse(tes)
            elif state[i,j]<lista[j].low:
                tes=lista[j].low+np.random.rand()*0.001
                print("start below lower bound, rounding up to: ",tes)
                temp[i,j]=lista[j].inverse(tes)
            else:
                temp[i,j]=lista[j].inverse(state[i,j])
    return temp


def untransform(state,lista):
    temp=np.zeros(state.shape)
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            temp[i,j]=lista[j](state[i,j])
    return temp
