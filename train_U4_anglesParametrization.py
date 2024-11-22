
#%%
from jax import random, grad, vmap, jit, lax
import jax.numpy as jnp
from jax.numpy import pi, cos, sin, arccos, arctan2, exp, array, mod, sum, zeros, mean, equal
from jax.example_libraries import stax, optimizers
from jax.tree_util import tree_flatten,tree_map
import random as rd
import pickle
from time import localtime,  strftime
strTime = strftime("%d_%H%M", localtime())


#%%
random_key, subkey = random.split(random.key(rd.getrandbits(32)))
layer_sizes = [2,250,350,150,4]
batches_size = [128 for i in range(401)]
learning_rate = 1.5e-4
l2regularizer = 2e-5
T_steps = 60


#%%
delta = 4*pi/T_steps

layers = []
for i in range(1,len(layer_sizes)):
    layers.append(stax.Dense(layer_sizes[i]))
    layers.append(stax.Relu)
layers.pop()
layers.append(stax.LogSoftmax)
init_random_params, apply_model = stax.serial(*layers)
_, params = init_random_params(subkey, (-1, T_steps, layer_sizes[0]))

opt_init, opt_update, opt_get_params = optimizers.adam(learning_rate)
opt_state = opt_init(params)
jit_update = jit(opt_update)
jit_get_params = jit(opt_get_params)


#%%
def xyz2tp(rd3Dcoord):
    return array([ arccos(rd3Dcoord[2]/jnp.linalg.norm(rd3Dcoord)),
                    pi+arctan2(-rd3Dcoord[1],-rd3Dcoord[0]) ])

def parametrize(tp):
    theta = tp[0]
    phi = tp[1]
    return array([ theta/pi , phi/pi/2 ])

def MC_sampling_action_from_input(key, proba): 
    return random.categorical(key,proba)

def step(tp,a):
    # +X, +Y, +Z, Id
    new_tp = array([
        equal(a,3)*tp[0]+
        equal(a,2)*tp[0]+
        equal(a,0)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*sin(tp[1])*sin(delta)) +
        equal(a,1)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*cos(tp[1])*sin(delta)) 
        ,
        equal(a,3)*tp[1]+
        equal(a,2)*mod(tp[1]+delta,2*pi)+
        equal(a,0)*( pi+arctan2(cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])))  +
        equal(a,1)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),-cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)))
    ])
    return new_tp , (1.0+cos(new_tp[0]))/2

batch_xyz2tp = jit(vmap(xyz2tp,in_axes=0,out_axes=0))
batch_parametrize = jit(vmap(parametrize, in_axes=0, out_axes=0))
batch_sampling = jit(vmap(MC_sampling_action_from_input, in_axes=(0,0), out_axes=0))
batch_greedySample = vmap(jnp.argmax,in_axes=0,out_axes=0)
batch_step = jit(vmap(step, in_axes=(0,0), out_axes=(0,0)))


#%%
def reset_env(random_key, batch_size):
    random_key, subkey = random.split(random_key)
    batch_ThetaPhi = batch_xyz2tp(random.normal(subkey,(batch_size,3)))
    """
    thetas = random.uniform(subkey,(batch_size,1))*pi
    random_key, subkey = random.split(random_key)
    phis = random.uniform(subkey,(batch_size,1))*2*pi
    batch_ThetaPhi = jnp.concatenate([thetas, phis], axis=1)
    """
    inputs_t = zeros((batch_size, T_steps, 2),dtype=jnp.float32)
    actionTypes_t = zeros((batch_size, T_steps),dtype=jnp.int32)
    rewards_t = zeros((batch_size, T_steps),dtype=jnp.float32)
    return random_key, batch_ThetaPhi, inputs_t, actionTypes_t, rewards_t

@jit
def jit_for_t_body(t:int, vals:tuple):

    random_key, batch_ThetaPhi, inputs_t, actionTypes_t, rewards_t = vals

    inputs = batch_parametrize(batch_ThetaPhi)
    inputs_t = inputs_t.at[:,t,:].set(inputs)

    random_key, subkey = random.split(random_key)
    actionTypes = batch_sampling( random.split(subkey, rewards_t.shape[0]) , exp(apply_model(params,inputs)) )
    actionTypes_t = actionTypes_t.at[:,t].set(actionTypes)

    new_batch_ThetaPhi, rewards = batch_step(batch_ThetaPhi, actionTypes)
    rewards_t = rewards_t.at[:,t].set(rewards)

    return (random_key, new_batch_ThetaPhi, inputs_t, actionTypes_t, rewards_t)

@jit
def pseudoloss(params2grad, inputs_t, actionTypes_t, returns_t):
    return (
                l2regularizer*
                sum(array(
                    tree_map(lambda x: sum(x**2), tree_flatten(params2grad)[0]) 
                ))
                - 
                mean(
                    sum(
                        jnp.take_along_axis(
                            apply_model(params2grad,inputs_t),
                            jnp.expand_dims(actionTypes_t,axis=2),
                        axis=2).squeeze()
                        *
                        ( returns_t - mean(returns_t, axis=0) ),
                    axis=1)
                )
            )


#%%
print_every_cycles = 1
test_batches = 500
fidelities = [[],[],[],[],[]]


for epoch in range(len(batches_size)):
    
    random_key, _, inputs_t, actionTypes_t, rewards_t = lax.fori_loop(0,T_steps, 
                            jit_for_t_body,( reset_env(random_key, batches_size[epoch]) ))
    cumulaRewards_t = jnp.flip(jnp.cumsum(jnp.flip(rewards_t,axis=1), axis=1),axis=1)
    opt_state = jit_update(epoch, grad(pseudoloss)(params, inputs_t, actionTypes_t, 
                            cumulaRewards_t), opt_state)
    params = jit_get_params(opt_state)
    
    ### print(jnp.mean(cumulaRewards_t[:,-1]),jnp.std(cumulaRewards_t[:,-1]),jnp.min(cumulaRewards_t[:,-1]))
    
    if print_every_cycles!=False and mod(epoch,print_every_cycles)==0: 
        
        #random_key, Tbatch_ThetaPhi, _, _, _ = reset_env(random_key, test_batches) 
        random_key, subkey = random.split(random_key)
        Tbatch_ThetaPhi = batch_xyz2tp(random.normal(subkey,(test_batches,3)))
        Tbatch_logproba = apply_model(params, batch_parametrize(Tbatch_ThetaPhi))
        Tentropy = mean( - exp(Tbatch_logproba) * Tbatch_logproba ) * 4
        print("Epoch: {:04d}, avg entropy: {:0.3f}".format(epoch, Tentropy), end="")

        for t in range(T_steps):
            Tacts = batch_greedySample( apply_model(params, batch_parametrize(Tbatch_ThetaPhi)) )
            Tbatch_ThetaPhi, Trewards = batch_step(Tbatch_ThetaPhi,Tacts)
        Tmen, Tsd, Tmn, Tmx = jnp.mean(Trewards), jnp.std(Trewards), jnp.min(Trewards), jnp.max(Trewards)
        fidelities[0].append(Tmen)
        fidelities[1].append(Tsd)
        fidelities[2].append(Tmn)
        fidelities[3].append(Tmx)
        fidelities[4].append(Tentropy)
        pickle.dump( (layer_sizes, batches_size, learning_rate, l2regularizer, T_steps, optimizers.unpack_optimizer_state(opt_state), fidelities),  open("./NNparams_"+strTime+".pkl", "wb"))
        print(" | mean: {:0.3f} , std: {:0.3f} , min: {:0.3f}".format(Tmen, Tsd, Tmn, Tmx))

        # if men-sd>0.98 or mn>0.95 : pickle.dump(optimizers.unpack_optimizer_state(opt_state),  open("./NNparams/"+strTime+"/"+str(epoch)+".pkl", "wb"))



#%%   