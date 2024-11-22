# define here the deep neural network representation of the policy
### jax.example_libraries.stax.Dense()
from env_batched import *
from jax.example_libraries import stax, optimizers
from jax.tree_util import tree_flatten,tree_map
#import numpy as np # shouldn't be used after debugging phase
import random as rd # used only once per NN

#%%
# a parametrization which can avoid the singularity in spherical angles' poles

def parametrize(input):

	"""
	output: normalized to [0~1] theta and phi
	"""

	
	def tp2xy(theta,phi):
		phi = mod(phi,pi/2)
		x = theta/pi + 1/2 - 1/pi*arccos((4*phi/pi-1)*sin(theta))
		y = theta/pi - 1/2 + 1/pi*arccos((4*phi/pi-1)*sin(theta))
		return array([x,y])

	
	def normalized_tp(theta,phi):
		return array([theta/pi, phi/2/pi])
	
	#return normalized_tp( input[0], input[1])
	return tp2xy( input[0], input[1])


def unparametrize(input):
	"""
	in uninvertible scenario, shouldn't be used!
	"""

	def unnormalized_tp(theta_normed, phi_normed):
		return array([pi*theta_normed, 2*pi*phi_normed])


	def xy2tp(x,y):
		theta = pi/2*(x+y)
		phi = pi/4*(1+sin(pi/2*(x-y))/sin(pi/2*(x+y)))
		return array([theta,phi])
	
	return xy2tp( input[0], input[1])


def evaluate_trajectory(tp, actions):
	"""
	Return final fidelity while taking trajectory of unitaries and applies them on the initial_state
	"""
	T = len(actions)
	bSphr = qt.Bloch()
	v2add = zeros((T+1,3),dtype=jnp.float32)
	v2add = v2add.at[0,:].set(array([sin(tp[0])*cos(tp[1]) , sin(tp[0])*sin(tp[1]) , cos(tp[0])]))

	for t in range(T):
		tp, reward = step(tp,actions[t])
		v2add = v2add.at[t+1,:].set(array([sin(tp[0])*cos(tp[1]) , sin(tp[0])*sin(tp[1]) , cos(tp[0])]))
	
	cmap = matplotlib.pyplot.get_cmap('inferno', T+1)    # PiYG
	bSphr.add_vectors(v2add)
	bSphr.vector_color = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
	bSphr.vector_width = 2
	bSphr.vector_alpha = [0.2+0.3/T*i for i in range(T+1)]
	bSphr.show()

	return reward


def most_proba_action(logProba): 
	return jnp.argmax(logProba)


class Policy():

	def __init__(self,random_key=random.key(rd.getrandbits(32)), layer_sizes=[2,800,192,7],
			  		  learning_rate=2e-4, batch_size=[128 for i in range(2001)], T_steps = 60 ):
		self.random_key = random_key
		self.layers_size = None
		self.params = None
		self.apply_model = None
		self.opt_state = None
		self.opt_update = None
		self.opt_get_params = None
		self.batch_size = batch_size
		self.T_steps = T_steps
		self._architecture(layer_sizes)
		self._optimizer(learning_rate)
		

	def _architecture(self, layer_sizes):
		"""
		contains the NN architecture

		Parameters:
			layers_size : (int) , the N_neurons in each layer,
			noted that the 1st layer size = 2, and last = 7,
			corresponding to 1/4-sphere parametrization & 7 actions.

		Return:
			gaussian randomly distributed NN params: [ ([w],[b]), ([w],[b]), ([w],[b])... ]
		"""
		
		layers = []
		for i in range(len(layer_sizes) - 1):
			layers.append(stax.Dense(layer_sizes[i + 1]))
			layers.append(stax.Relu)
		layers.pop()
		layers.append(stax.LogSoftmax)
		init_random_params, self.predict = stax.serial(*layers)
		self.random_key, subkey = random.split(self.random_key)
		_, self.params = init_random_params(subkey, (-1, layer_sizes[0]))


	def _optimizer(self, learning_rate):
		"""
		initiate the deep learning optimizer: Adam
		"""
		opt_init, self.opt_update, self.opt_get_params = optimizers.adam(learning_rate)
		self.opt_state = opt_init(self.params)


	def batch_collect_traj(self,batch_size):
		
		inputs_t = zeros((batch_size, self.T_steps, 2),dtype=jnp.float32)
		actionTypes_t = zeros((batch_size, self.T_steps),dtype=jnp.int32)
		rewards_t = zeros((batch_size, self.T_steps),dtype=jnp.float32)
		self.random_key, subkey = random.split(self.random_key)
		env = QubitEnv(self.T_steps, batch_size, subkey)

		batch_step = vmap(step, in_axes=(0,0), out_axes=(0,0) )
		def MC_sampling_action(keys, logProba): return random.categorical(keys,exp(logProba))
		batch_sampling = vmap( MC_sampling_action, in_axes=(0,0), out_axes=0 )
		batch_parametriz =  vmap(parametrize, in_axes=0, out_axes=0) 

		for t in range(self.T_steps):

			inputs = batch_parametriz(env.batch_ThetaPhi)
			inputs_t = inputs_t.at[:,t,:].set(inputs)

			self.random_key, subkey = random.split(self.random_key)
			actionTypes = batch_sampling( random.split(subkey, batch_size), self.predict(self.params, inputs) ) # the apply_model part is purely functional, so jittable
			actionTypes_t = actionTypes_t.at[:,t].set(actionTypes)

			env.batch_ThetaPhi, rewards = batch_step(env.batch_ThetaPhi, actionTypes)
			rewards_t = rewards_t.at[:,t].set( rewards )

		return inputs_t, actionTypes_t, rewards_t
	

	def batch_update_params(self,epoch):

		inputs_t, actionTypes_t, rewards_t = self.batch_collect_traj(self.batch_size[epoch])
		returns_t = jnp.flip(jnp.cumsum(jnp.flip(rewards_t,axis=1), axis=1),axis=1)
		#returns_t = jnp.cumsum(rewards_t[:,::-1], axis=1)[:,::-1]
		baseline_t = mean(returns_t, axis=0)
		#print(returns_t-baseline_t)
		def batch_pseudoloss_functional(params, inputs_t, actionTypes_t,returns_t,baseline_t ):
			"""
			TODO: try jit with pseudoloss and jax.grad; or just do them inside policy_gradient
			"""
			logProbas_t_a = self.predict(params,inputs_t)
			#logProbas_t = (logProbas_t_a[jnp.arange(7)==actionTypes_t[...,None]]).reshape(self.batch_size[epoch],self.T_steps)
			logProbas_t = jnp.take_along_axis(logProbas_t_a,jnp.expand_dims(actionTypes_t,axis=2),axis=2).squeeze()
			return -mean(sum(logProbas_t*(returns_t-baseline_t),axis=1)) + 1e-4*sum(array(tree_map(lambda x: sum(x**2), tree_flatten(params)[0])))
		
		grads = jax.grad(batch_pseudoloss_functional)(self.params, inputs_t, actionTypes_t,returns_t,baseline_t)
		self.opt_state = self.opt_update(epoch, grads, self.opt_state)
		self.params = self.opt_get_params(self.opt_state)












