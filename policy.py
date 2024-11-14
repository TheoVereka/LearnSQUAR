# define here the deep neural network representation of the policy
### jax.example_libraries.stax.Dense()
from env import *
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.example_libraries import stax, optimizers
import jax
import random as rd # used only once per NN

#%%
# a parametrization which can avoid the singularity in spherical angles' poles

def parametrize(state:Qubit):

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


class Policy():

	def __init__(self,random_key=random.key(rd.getrandbits(32)),layer_sizes=[2,10,30,20,7],learning_rate=1e-3):
		self.random_key = random_key
		self.layers_size = None
		self.activation_func = stax.ReLu
		self.params = None
		self.apply_model = None
		self.opt_state = None
		self.opt_update = None
		self.opt_get_params = None
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
			layers.append(self.activation_func)
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





	def predict(self,initial_state:Qubit):
		"""
		evaluate the policy to generate an action_type by
		their probability normalized by logsoftmax function 
		"""
		inputs = parametrize(initial_state)
		return (self.apply_model)(self.params, inputs)	
	

	def MC_sampling_action(self,logProba):
		proba_action = exp(logProba)
		self.random_key, subkey = random.split(self.random_key)
		action_type = int(random.choice(subkey,7,p=proba_action))-3
		return action_type
	

	def most_proba_action(self,logProba):
		action_type = jnp.argmax(logProba)-3
		return action_type
	




	def collect_traj(self, total_steps:int, batch_size:int):
		"""
		add sth.
		"""
		env = QubitEnv(total_steps)
		env.reset()
		trajectories_type = []
		initial_states = []
		# gradient[eta] = 0

		for i in range(batch_size): 								### use vmap to do auto-batching
			self.random_key, subkey = random.split(self.random_key)				
			env.reset(subkey)
			trajectories_type.append([])
			initial_states.append(env.state)
			total_reward = 0
			# grad_batch[eta] = 0

			for t in range(total_steps):
				action_type = self.predict(env.state)				### use vmap to do auto-batching
				trajectories_type[i].append(action_type)
				# grad_batch[eta] += autodifferation_of_log_pi(action_type|state)
				total_reward += env.step(action_type)
			#  grandient[eta]+= total_reward * grad_batch[eta]

		# gradient[eta] /= batch_size

		return initial_states, trajectories_type

	
	def pseudoloss_functional(proba, states, trajs):
		return -expected
	

	def update_params(self,epoch):
		"""
		compute the gradients of the policy w.r.t. the NN parameters
		Parameter:
			input : current state labeled in normalized (x,y)
		"""
		grads = jax.grad(self.pseudoloss_functional)(self.params)
		self.opt_state = self.opt_update(epoch, grads, self.opt_state)
		self.params = self.opt_get_params(self.opt_state)
