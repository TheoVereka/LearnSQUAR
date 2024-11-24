#%%
import jax.numpy as jnp
from jax.numpy import pi, cos, sin, arccos, arctan2, exp, array, mod, sum, zeros, mean, equal
from jax.example_libraries import stax, optimizers
import pickle
import matplotlib.pyplot as plt
import qutip
import numpy.random as nprd

#%%
strTime = "24_1521"

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

fig = plt.figure(dpi=300)
ax = fig.add_subplot()
#ax.set_xscale("log")
#ax.set_yscale("log")
ax.loglog([1-elem for elem in fidelitieS[0]],label="Mean infidelity",alpha=0.90)
ax.loglog([elem for elem in fidelitieS[1]],label="Standard error",alpha=0.8)
ax.loglog([1-elem for elem in fidelitieS[2]],label="Maximum infidelity",alpha=0.95)
ax2 = ax.twinx() 
ax2.loglog([elem/jnp.log(2) for elem in fidelitieS[4]],label="Averaged entropy",color="red",alpha=0.75)
ax.set_xlim((1,1e4))
ax.set_ylim((0.002,4))
ax.set_xlabel("Episodes (1)")
ax.set_ylabel("Infidelity = 1-|<ψ|target>|² (1)")
ax2.set_ylabel("Entropy averaged on sphere (bit)")
ax2.set_ylim((0.002,4))
for item in ([ax.xaxis.label, ax.yaxis.label,ax2.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()+ax2.get_yticklabels()):
    item.set_fontsize(14)
ax.grid(visible=True, which="major",axis="both")
ax.grid(visible=True, which="minor",axis="both",alpha=0.15)
fig.legend(loc=(0.16,0.2))
fig.show()

"""
fig = plt.figure(dpi=300)
ax = fig.add_subplot()
ax.plot(jnp.log(1-array(fidelitieS[0])))
ax.plot(jnp.log(1-array(fidelitieS[2])))
fig.show()
"""

print(fidelitieS[0][-1],fidelitieS[2][-1])


#%%
b = qutip.Bloch()
#b.render()
number_line = 15
points = [[],[],[]]
for i in range(number_line+1):
     for j in range(number_line+1):
            input = array([i/number_line+(5-i)/10000,(j/number_line+(5-j)/10000)/2+1/4])
            #input = nprd.rand(2)
            tp = unparametrize(input)
            point1 = tp2xyz(tp)
            a_rot = round(jnp.argmax(apply_model(params,input)))
            delta= delta/2
            new_tp = array([
        jnp.greater_equal(a_rot,4)*tp[0]+
        jnp.less(a_rot,4)*(
            equal(a_rot,3)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*cos(tp[1])*sin(delta)) +
            equal(a_rot,2)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*sin(tp[1])*sin(delta)) +
            equal(a_rot,0)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*sin(tp[1])*sin(delta)) +
            equal(a_rot,1)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*cos(tp[1])*sin(delta)) )
        ,
        equal(a_rot,6)*tp[1]+
        equal(a_rot,4)*mod(tp[1]+delta,2*pi)+
        equal(a_rot,5)*mod(tp[1]-delta,2*pi)+
        jnp.less(a_rot,4)*(
            equal(a_rot,3)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)))  +
            equal(a_rot,2)*( pi+arctan2(-cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1]))) +
            equal(a_rot,0)*( pi+arctan2(cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])))  +
            equal(a_rot,1)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),-cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta))) )
        ])
            delta= delta*2
            point2 = tp2xyz(new_tp+array([0.00001,0.00001]))
            b.add_arc( start=point1, end=point2, fmt='r' )
            points[0].append(point1[0])
            points[1].append(point1[1])
            points[2].append(point1[2])
b.add_points(points,alpha=0.5)
b.add_vectors([1,0,0],alpha=0.67)
b.add_vectors([0,1,0])
b.add_vectors([0,0,1])
#b.view = [110,-30]
b.show()


if False:
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

#%%