
#%%
import jax
from jax import random, grad, vmap, jit, lax
import jax.numpy as jnp
from jax.numpy import pi, cos, sin, arccos, arctan2, exp, array, mod, sum, zeros, mean, equal
from jax.example_libraries import stax, optimizers
from jax.tree_util import tree_flatten,tree_map
import random as rd
import pickle
from time import localtime,  strftime

layer_sizes = [2,250,150,103,7]

#%%
for meow in [16,32,64,128,256,512]:
    strTime = strftime("%d_%H%M%S", localtime())
    random_key, subkey = random.split(random.key(rd.getrandbits(32)))
    batches_size = [meow for i in range(9001)]
    learning_rate = 1e-3
    l2regularizer = 5e-6
    T_steps = 60


    #%%
    delta = 4*pi/T_steps

    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(stax.Dense(layer_sizes[i + 1]))
        layers.append(stax.Relu)
    layers.pop()
    layers.append(stax.LogSoftmax)
    init_random_params, apply_model = stax.serial(*layers)
    _, params = init_random_params(subkey, (-1, layer_sizes[0]))

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
        return array([ theta/pi ,phi/pi/2 ])

    def MC_sampling_action_from_input(key, input): 
        """ WARNING: *params* is defined GLOBALLY to accelerate the speed! """
        return random.categorical(key,apply_model(params, input))

    def step(tp,a):
        new_tp = array([
            jnp.greater_equal(a,4)*tp[0]+
            (
                equal(a,3)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*cos(tp[1])*sin(delta)) +
                equal(a,2)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*sin(tp[1])*sin(delta)) +
                equal(a,0)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*sin(tp[1])*sin(delta)) +
                equal(a,1)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*cos(tp[1])*sin(delta)) )
            ,
            equal(a,6)*tp[1]+
            equal(a,4)*mod(tp[1]+delta,2*pi)+
            equal(a,5)*mod(tp[1]-delta,2*pi)+
            (
                equal(a,3)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)))  +
                equal(a,2)*( pi+arctan2(-cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1]))) +
                equal(a,0)*( pi+arctan2(cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])))  +
                equal(a,1)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),-cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta))) )
        ])
        return new_tp ,(1+cos(new_tp[0]))/2

    batch_xyz2tp = jit(vmap(xyz2tp,in_axes=0,out_axes=0))
    batch_parametrize = jit(vmap(parametrize, in_axes=0, out_axes=0))
    batch_sampling = vmap(MC_sampling_action_from_input, in_axes=(0,0), out_axes=0)
    batch_greedySample = vmap(jnp.argmax,in_axes=0,out_axes=0)
    batch_step = jit(vmap(step, in_axes=(0,0), out_axes=(0,0)))


    #%%
    def reset_env(random_key, batch_size):
        random_key, subkey = random.split(random_key)
        batch_ThetaPhi = batch_xyz2tp(random.normal(subkey,(batch_size,3)))
        inputs_t = zeros((batch_size, T_steps, 2),dtype=jnp.float32)
        actionTypes_t = zeros((batch_size, T_steps),dtype=jnp.int32)
        rewards_t = zeros((batch_size, T_steps),dtype=jnp.float32)
        return random_key, batch_ThetaPhi, inputs_t, actionTypes_t, rewards_t

    """
    @jit
    def jit_for_t_body(t:int, vals:tuple):

        random_key, batch_ThetaPhi, inputs_t, actionTypes_t, rewards_t = vals

        inputs = batch_parametrize(batch_ThetaPhi)
        inputs_t = inputs_t.at[:,t,:].set(inputs)

        random_key, subkey = random.split(random_key)
        actionTypes = batch_sampling( random.split(subkey, rewards_t.shape[0]) , inputs )
        actionTypes_t = actionTypes_t.at[:,t].set(actionTypes)

        batch_ThetaPhi, rewards = batch_step(batch_ThetaPhi, actionTypes)
        rewards_t = rewards_t.at[:,t].set(rewards)

        return (random_key, batch_ThetaPhi, inputs_t, actionTypes_t, rewards_t)
    """

    @jit
    def pseudoloss(params2grad, inputs_t, actionTypes_t, returns_t):
        return ( 
                    - mean(
                        sum(
                            jnp.take_along_axis(
                                apply_model(params2grad,inputs_t),
                                jnp.expand_dims(actionTypes_t,axis=2),
                            axis=2).squeeze()
                            *
                            ( returns_t - mean(returns_t, axis=0) ),
                        axis=1)
                    )
                    + 
                    l2regularizer*
                    sum(array(
                        tree_map(lambda x: sum(x**2), tree_flatten(params2grad)[0]) 
                    ))
                )


    #%%
    print_every_cycles = 1
    test_batches = 512
    fidelities = [[],[],[],[],[]]
    record_entropy = 0.4
    record_mean = 0.9925
    record_mean_std = 0.99
    record_mn = 0.985


    for epoch in range(len(batches_size)):
        
        #random_key, _, inputs_t, actionTypes_t, rewards_t = lax.fori_loop(0,T_steps, 
        #                        jit_for_t_body,( reset_env(random_key, batches_size[epoch]) ))
        #"""
        random_key, batch_ThetaPhi, inputs_t, actionTypes_t, rewards_t = reset_env(random_key, batches_size[epoch])
        for t in range(T_steps):
            inputs = batch_parametrize(batch_ThetaPhi)
            inputs_t = inputs_t.at[:,t,:].set(inputs)

            random_key, subkey = random.split(random_key)
            actionTypes = batch_sampling( random.split(subkey, rewards_t.shape[0]) , inputs )
            actionTypes_t = actionTypes_t.at[:,t].set(actionTypes)
            
            batch_ThetaPhi, _ = batch_step(batch_ThetaPhi, actionTypes)
            rewards_t = rewards_t.at[:,t].set(1-batch_ThetaPhi[:,0]/pi)
        #"""
        
        opt_state = jit_update(epoch, grad(pseudoloss)(params, inputs_t, actionTypes_t, 
                                jnp.flip(jnp.cumsum(jnp.flip( rewards_t ,axis=1), axis=1),axis=1)), opt_state)
        params = jit_get_params(opt_state)

        if print_every_cycles!=False and mod(epoch,print_every_cycles)==0: 
            
            random_key, test_ThetaPhi, _, _, _ = reset_env(random_key, test_batches) 
            batch_logproba = apply_model(params, batch_parametrize(test_ThetaPhi))
            entropy = mean( - exp(batch_logproba) * batch_logproba ) * layer_sizes[-1]
            print("Epoch: {:04d}, avg entropy: {:0.3f}".format(epoch, entropy), end="")

            for t in range(T_steps):
                acts = batch_greedySample( apply_model(params, batch_parametrize(test_ThetaPhi)) )
                test_ThetaPhi, Fid = batch_step(test_ThetaPhi,acts)
            men, sd, mn, mx = jnp.mean(Fid), jnp.std(Fid), jnp.min(Fid), jnp.max(Fid)
            fidelities[0].append(men)
            fidelities[1].append(sd)
            fidelities[2].append(mn)
            fidelities[3].append(mx)
            fidelities[4].append(entropy)
            pickle.dump( (layer_sizes, batches_size, learning_rate, l2regularizer, T_steps, optimizers.unpack_optimizer_state(opt_state), fidelities),  open("./NNparams_"+strTime+".pkl", "wb"))
            print(" | mean: {:0.3f} , std: {:0.3f} , min: {:0.3f}".format(men, sd, mn, mx))
            if entropy<record_entropy and men>record_mean and (mn>record_mn or men-sd>record_mean_std):
                record_mean = men
                record_mean_std = max(record_mean_std,men-sd)
                record_mn = max(mn, record_mn)
                pickle.dump( (layer_sizes, batches_size, learning_rate, l2regularizer, T_steps, optimizers.unpack_optimizer_state(opt_state), fidelities),  open("./NNparams_"+strTime+"_max.pkl", "wb"))



    #%%   