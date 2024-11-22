
#%%
from policy_batched import *
import pickle
from time import localtime, strftime
strTime = strftime("%d_%H%M", localtime())

train_with_show = False
print_every_cycles = 10 # round(len(myPo.batch_size)/200+0.5)
test_batches = 256

#%%
greedySample = vmap(most_proba_action,in_axes=0,out_axes=0)
vParam = vmap(parametrize,in_axes=0, out_axes=0)
vStep = vmap(step,in_axes=(0,0),out_axes=(0,0))
if train_with_show: 

    myPo = Policy()
    for epoch in range(len(myPo.batch_size)):

        if mod(epoch,print_every_cycles)==0: 
            print(epoch)
            pickle.dump(optimizers.unpack_optimizer_state(myPo.opt_state),  open("./NNparams/c"+strTime+".pkl", "wb"))

            myPo.random_key,subkey = random.split(myPo.random_key)
            myEnv = QubitEnv(myPo.T_steps,test_batches,subkey)
            randind = random.randint(myPo.random_key,1,0,test_batches)
            tp_initial = myEnv.batch_ThetaPhi[randind][0]
            print("init avgXYZ: {:0.3f} {:0.3f} {:0.3f}, init θ,φ = {:0.3f}, {:0.3f}".format(
                mean(sin(myEnv.batch_ThetaPhi[:,0])*cos(myEnv.batch_ThetaPhi[:,1])), 
                mean(sin(myEnv.batch_ThetaPhi[:,0])*sin(myEnv.batch_ThetaPhi[:,1])), 
                mean(cos(myEnv.batch_ThetaPhi[:,0])), tp_initial[0], tp_initial[1] ) )
            actionType_t = zeros((test_batches,myPo.T_steps),dtype=jnp.int32)
            for t in range(myPo.T_steps):
                acts = greedySample( myPo.predict(myPo.params, vParam(myEnv.batch_ThetaPhi)) ) # the apply_model part is purely functional, so jittable
                actionType_t = actionType_t.at[:,t].set(acts)
                myEnv.batch_ThetaPhi, rewards = vStep(myEnv.batch_ThetaPhi,acts)
            _ = evaluate_trajectory(tp_initial, actionType_t[randind][0])
            final_rewards = rewards
            print("mean: {:0.3f}  std: {:0.3f}  min: {:0.3f} max: {:0.3f}\n".format(
                jnp.mean(final_rewards), jnp.std(final_rewards), jnp.min(final_rewards), jnp.max(final_rewards) ) )
        
        myPo.batch_update_params(epoch)

    pickle.dump(optimizers.unpack_optimizer_state(myPo.opt_state),  open("./NNparams/c"+strTime+".pkl", "wb"))

else:
    myPo = Policy()
    for epoch in range(len(myPo.batch_size)):
        fidelities = [[],[],[],[]]
        myPo.batch_update_params(epoch)
        if mod(epoch,print_every_cycles)==0: 
            print(epoch)
            pickle.dump(optimizers.unpack_optimizer_state(myPo.opt_state),  open("./NNparams/"+strTime+"/"+str(epoch)+".pkl", "wb"))
            myPo.random_key,subkey = random.split(myPo.random_key)
            myEnv = QubitEnv(myPo.T_steps,test_batches,subkey)
            for t in range(myPo.T_steps):
                acts = greedySample( myPo.predict(myPo.params, vParam(myEnv.batch_ThetaPhi)) )
                myEnv.batch_ThetaPhi, rewards = vStep(myEnv.batch_ThetaPhi,acts)
            men, sd, mn, mx = jnp.mean(rewards), jnp.std(rewards), jnp.min(rewards), jnp.max(rewards)
            fidelities[0].append(men)
            fidelities[1].append(sd)
            fidelities[2].append(mn)
            fidelities[3].append(mx)
            print("mean: {:0.4f} , std: {:0.4f} , min: {:0.4f}, max: {:0.4f}".format(men, sd, mn, mx))
    pickle.dump(fidelities,  open("./NNparams/"+strTime+"/fidelities_records.pkl", "wb"))
             
# %%
