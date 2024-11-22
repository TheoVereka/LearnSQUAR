#%%
import jax.numpy as jnp
from jax.numpy import pi, cos, sin, arccos, arctan2, exp, array, mod, sum, zeros, mean, equal
from jax.example_libraries import stax, optimizers
import pickle
import matplotlib.pyplot as plt
import qutip
import numpy.random as nprd

#%%
strTime = "21_2330"

#%%
layer_sizes, batches_size, learning_rate, l2regularizer, T_steps, load_state, fidelitieS = pickle.load(open("./NNparams_"+strTime+".pkl", "rb"))
print(layer_sizes)
print(batches_size[0],learning_rate,l2regularizer)
delta = 4*pi/T_steps
layers = []
for i in range(len(layer_sizes) - 1):
    layers.append(stax.Dense(layer_sizes[i + 1]))
    layers.append(stax.Relu)
layers.pop()
layers.append(stax.LogSoftmax)
init_random_params, apply_model = stax.serial(*layers)
opt_init, opt_update, opt_get_params = optimizers.adam(learning_rate)
params = opt_get_params(optimizers.pack_optimizer_state(load_state))

def parametrize(tp):
    theta = tp[0]
    phi = mod(tp[1],pi/2)
    return array([ theta/pi, phi/pi/2 ])

def unparametrize(xy):
    return array([ xy[0]*pi , xy[1]*2*pi ])

def tp2xyz(tp):
	return array([sin(tp[0])*cos(tp[1]) , sin(tp[0])*sin(tp[1]) , cos(tp[0])])

fig = plt.figure()
ax = fig.add_subplot()
for i in range(5):
    ax.plot(fidelitieS[i])



#%%
b = qutip.Bloch()
#b.render()
number_line = 20
points = [[],[],[]]
for i in range(number_line+1):
     for j in range(number_line+1):
            input = array([i/number_line+(5-i)/10000,j/number_line+(5-j)/10000])
            #input = nprd.rand(2)
            tp = unparametrize(input)
            point1 = tp2xyz(tp)
            a_rot = round(jnp.argmax(apply_model(params,input))) # *0+1
            delta= delta/2
            # +X, +Y, +Z, Id
            new_tp = array([
                equal(a_rot,3)*tp[0]+
                equal(a_rot,2)*tp[0]+
                equal(a_rot,0)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*sin(tp[1])*sin(delta)) +
                equal(a_rot,1)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*cos(tp[1])*sin(delta)) 
                ,
                equal(a_rot,3)*tp[1]+
                equal(a_rot,2)*mod(tp[1]+delta,2*pi)+
                equal(a_rot,0)*( pi+arctan2(cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])))  +
                equal(a_rot,1)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),-cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)))
            ])
            delta= delta*2
            point2 = tp2xyz(new_tp+array([0.00001,0.00001]))
            b.add_arc( start=point1, end=point2, fmt='r' )
            points[0].append(point1[0])
            points[1].append(point1[1])
            points[2].append(point1[2])
b.add_points(points,alpha=0.5)
#b.view = [-30,-30]
#b.view=[-180,0]
b.show()

#%%
b = qutip.Bloch()
#b.render()
number_line = 10
for i in range(number_line):
     for j in range(number_line):
            point1 = tp2xyz((unparametrize( (i/number_line+(5-i)/1000,j/number_line+(5-j)*1000) )))
            point2 = tp2xyz((unparametrize( ((i+1)/number_line+(4-i)/1000,(j)/number_line+(5-j)*1000) )))
            point3 = tp2xyz((unparametrize( ((i)/number_line+(5-i)/1000,(j+1)/number_line+(4-j)*1000) )))
            b.add_arc( start=point1, end=point2 )
            b.add_arc( start=point1, end=point3 )
b.add_arc(array([1,0,0]),array([0,0,-1]))
b.add_arc(array([0,1,0]),array([0,0,-1]))
b.show()