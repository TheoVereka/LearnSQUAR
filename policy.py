# define here the deep neural network representation of the policy
### jax.example_libraries.stax.Dense()
from env import *
import jax.numpy as jnp
from jax import grad, vmap
from jax.example_libraries import stax, optimizers
import jax
import random as rd # used only once per NN

#%%
# a parametrization which can avoid the singularity in spherical angles' poles

def parametrize(state:Qubit):

	"""
	output: normalized to [0~1] theta and phi
	"""

	theta, phi = state.ThetaPhi
	
	def tp2xy(theta,phi):
		phi = mod(phi,pi/2)
		x = theta/pi + 1/2 - 1/pi*arccos((4*phi/pi-1)*sin(theta))
		y = theta/pi - 1/2 + 1/pi*arccos((4*phi/pi-1)*sin(theta))
		return array([x,y])

	def xy2tp(x,y):
		theta = pi/2*(x+y)
		phi = pi/4*(1+sin(pi/2*(x-y))/sin(pi/2*(x+y)))
		return (theta,phi)
	
	def normalized_tp(theta,phi):
		return array([theta/pi, phi/2/pi])
	
	return normalized_tp(theta,phi)

def unparametrize(inputs):
	return pi*inputs[0], 2*pi*inputs[1]


class Policy():

	def __init__(self,random_key=random.key(rd.getrandbits(32)), layer_sizes=[2,10,20,7],
			  		  learning_rate=7e-2, batch_size=[5 for i in range(17)], T_steps = 13):
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
		init_random_params, self.apply_model = stax.serial(*layers)
		self.random_key, subkey = random.split(self.random_key)
		_, self.params = init_random_params(subkey, (-1, layer_sizes[0]))


	def _optimizer(self, learning_rate):
		"""
		initiate the deep learning optimizer: Adam
		"""
		opt_init, self.opt_update, self.opt_get_params = optimizers.adam(learning_rate)
		self.opt_state = opt_init(self.params)





	def predict(self,params, initial_state:Qubit):
		"""
		evaluate the policy to generate an action_type by
		their probability normalized by logsoftmax function 
		"""
		inputs = parametrize(initial_state)
		return (self.apply_model)(params, inputs)	
	
	def predict_inputs(self,params, inputs):
		"""
		evaluate the policy to generate an action_type by
		their probability normalized by logsoftmax function 
		"""
		return (self.apply_model)(params, inputs)	
	

	def MC_sampling_action(self,logProba):
		proba_action = exp(logProba)
		self.random_key, subkey = random.split(self.random_key)
		action_type = random.choice(subkey,7,p=proba_action)-3
		return action_type
	

	def most_proba_action(self,logProba):
		action_type = jnp.argmax(logProba)-3
		return action_type
	








	def batch_collect_traj(self,inputs):

		env = QubitEnv(self.T_steps)
		env.state = Qubit(theta=0,phi=0)
		env.state.ThetaPhi = unparametrize(inputs)
		action_types = []
		input_states = []
		rewards = []

		for t in range(self.T_steps):
			inputs=parametrize(env.state)
			input_states.append(inputs)
			action_type = self.MC_sampling_action(self.predict_inputs(self.params, inputs))
			print(action_type)
			action_types.append(action_type)
			state, reward, done = env.step(action_type)
			rewards.append(reward)

		return array(input_states), array(action_types), array(rewards)
	

	def batch_update_params(self,epoch):

		batched_collect_traj = vmap(self.batch_collect_traj)

		def batch_pseudoloss_functional(params):
			
			random_inputs = []
			for batch in range(self.batch_size[epoch]):
				self.random_key, subkey = random.split(self.random_key)
				random_inputs.append(parametrize(Qubit(subkey)))
			inputsS = jnp.stack(random_inputs)
			
			pseudo_loss = 0
			statesS, actionsS, rewardsS = batched_collect_traj(inputsS) # 如何vectorize applying action
			print()
			print(statesS)
			print()
			print(actionsS)
			print()
			print(rewardsS)

			pseudo_loss -= sum(array(logProbas)) * sum(array(rewards))
			return pseudo_loss/self.batch_size[epoch]
		
		grads = jax.grad(batch_pseudoloss_functional)(self.params)
		self.opt_state = self.opt_update(epoch, grads, self.opt_state)
		self.params = self.opt_get_params(self.opt_state)












	def collect_traj(self, total_steps:int,initial_state=None):
		"""
		add sth.
		"""
		env = QubitEnv(total_steps)
		if isinstance(initial_state,Qubit):
			env.state = initial_state
		else:
			self.random_key, subkey = random.split(self.random_key)				
			env.reset(subkey)
		action_types = []
		states = []
		rewards = []

		for t in range(total_steps):
			states.append(env.state)
			action_type = self.MC_sampling_action(self.predict(self.params, env.state))
			action_types.append(action_type)
			# grad_batch[eta] += autodifferation_of_log_pi(action_type|state)
			state, reward, done = env.step(action_type)
			rewards.append(reward)

		return states, action_types, rewards


	def update_params(self,epoch):
		"""
		compute the gradients of the policy w.r.t. the NN parameters
		Parameter:
			batch_size : (int), number of batched trajectories in this epoch
		"""
		def pseudoloss_functional(params):
			"""
			TO_DO: auto_batch: select the actions from each batch

			Reward_single_batch = SUM_t{ (predict(state[t]))[action[t]] } * SUM_t{ reward[t] } 
			total_pseudoloss = - E{Reward_single_batch} = E{loss_single_batch}
			"""
			pseudo_loss = 0
			for traj in range(self.batch_size[epoch]):
				states, actions, rewards = self.collect_traj(self.T_steps)
				logProbas = [ (self.predict(params,states[t]))[actions[t]+3] for t in range(self.T_steps)]
				pseudo_loss -= sum(array(logProbas)) * sum(array(rewards))
			pseudo_loss/=self.batch_size[epoch]
			return pseudo_loss
		
		grads = jax.grad(pseudoloss_functional)(self.params)
		self.opt_state = self.opt_update(epoch, grads, self.opt_state)
		self.params = self.opt_get_params(self.opt_state)