
# coding: utf-8

# In[411]:


import numpy as np
import pandas as pd
import random
np.random.seed(1000)
input_array=np.random.uniform(size=(50,1))
output_array=np.sin(input_array)
total_rows=input_array.shape[0]
# np.seterr(divide='ignore', invalid='ignore')


# In[412]:




# In[413]:


def tanh(x):
    t=np.tanh(x)
    return t


# In[414]:


def sigmoid(x):
    d=(1/(1+np.exp(-x)))
    return d


# In[415]:


def derivative_tanh(x):
    y=1.0 - np.tanh(x)**2
    return y


# In[407]:


def derivative_sigmoid(x):
    derivative=x*(1-x)
    return derivative
    


# In[420]:


epoch=20000
hidden_layer=3
input_neurons=1
output_neurons=1
learning_rate=0.1
input_array=data['input'].values.reshape(total_rows,1)
# input_array=input_array.astype(np.float64)
output_array=data['output'].values.reshape(total_rows,1)
# output_array=output_array.astype(np.float64)

weights_in=np.random.uniform(size=(input_neurons,hidden_layer)) 
# weights_in=weights_in.astype(np.float64)
bias_in=np.random.uniform(size=(1,hidden_layer))
# bias_in=bias_in.astype(np.float64)
weights_out=np.random.uniform(size=(hidden_layer,output_neurons))
# weights_out=weights_out.astype(np.float64)
bias_out=np.random.uniform(size=(1,output_neurons))
# bias=weights_in.astype(np.float64)

for i in range(epoch):

    #forward propogation
    hidden_layer_output=np.dot(input_array,weights_in)+bias_in
    activation_1=tanh(hidden_layer_output)
    activation_2_input=np.dot(activation_1,weights_out)+bias_out
    predicted_output=sigmoid(activation_2_input)


    # #backward propogation

    Error=(predicted_output-output_array)
    # print (Error)

    rate_change_output=derivative_sigmoid(predicted_output)
    rate_change_hidden_output=derivative_tanh(activation_1)
    error_on_output=Error*rate_change_output
    error_hidden_layer=error_on_output.dot(weights_out.T)
    delta_hidden_layer=error_hidden_layer*rate_change_hidden_output
    weights_out+=activation_1.T.dot(error_on_output)*learning_rate
    weights_in+=input_array.T.dot(delta_hidden_layer)*learning_rate
    bias_out+=np.sum(error_on_output,axis=0,keepdims=True)*learning_rate
    bias_in+=np.sum(error_hidden_layer,axis=0,keepdims=True)*learning_rate
print (predicted_output)


# In[402]:


import numpy as np

#Input array
X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

#Output
y=np.array([[1],[1],[0]])

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Variable initialization
epoch=5000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):
    #     #Forward Propogation
    hidden_layer_input1=np.dot(X,wh)
    hidden_layer_input=hidden_layer_input1 + bh
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,wout)
    output_layer_input= output_layer_input1+ bout
    output = sigmoid(output_layer_input)

    #Backpropagation
    E = y-output
    slope_output_layer = derivatives_sigmoid(output)
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(wout.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    wout += hiddenlayer_activations.T.dot(d_output) *lr
    bout += np.sum(d_output, axis=0,keepdims=True) *lr
    wh += X.T.dot(d_hiddenlayer) *lr
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr

print (Error)


# In[22]:




