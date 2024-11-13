# define here the deep neural network representation of the policy
from env import *
import jax.numpy as jnp
from jax import grad, jit, vmap
import jax
import random as rd # used only once per NN

#%%
# a parametrization which can avoid the singularity in spherical angles' poles
def tp2xy(theta,phi):
    phi = mod(phi,pi/2)
    x = theta/pi + 1/2 - 1/pi*arccos((4*phi/pi-1)*sin(theta))
    y = theta/pi - 1/2 + 1/pi*arccos((4*phi/pi-1)*sin(theta))
    return (x,y)

def xy2tp(x,y):
    theta = pi/2*(x+y)
    phi = pi/4*(1+sin(pi/2*(x-y))/sin(pi/2*(x+y)))
    return (theta,phi)


class Policy():

	def __init__(self,random_key=random.key(rd.getrandbits(32))):
		self.random_key = random_key
		self.layers_size = None
		self.weight_normalization = None
		self.activation_func = None
		self.params = None
		


	def _architecture(self, layers_size=[2,10,30,20,7],activation_func="ReLu"):
		"""
		contains the NN architecture

		Parameters:
			layers_size : (int) , the N_neurons in each layer,
			noted that the 1st layer size = 2, and last = 7,
			corresponding to 1/4-sphere parametrization & 7 actions.

		Return:
			gaussian randomly distributed NN params: [ ([w],[b]), ([w],[b]), ([w],[b])... ]
		"""
		def random_layer_params(prev_layer, next_layer, key, scale_biais=1e-2):
			w_key, b_key = random.split(key)
			return ( (self.weight_normalization)(prev_layer) * random.normal(w_key, (next_layer, prev_layer,)), 
		   			 scale_biais * random.normal(b_key, (next_layer,)) )

		def init_network_params(sizes):
			keys = random.split(self.random_key, len(sizes))
			return [random_layer_params(prev_layer, next_layer, key) 
		   			for prev_layer, next_layer, key in zip(sizes[:-1], sizes[1:], keys)]

		self.layers_size = layers_size
		match activation_func:
			case "ReLu" : 
				self.activation_func = (lambda outputs: jnp.maximum(0,outputs))
				self.weight_normalization = (lambda prev_layer: jnp.sqrt(2.0/prev_layer))
			case _ : print("Currently only support ReLu")
		self.params = init_network_params(layers_size)
		

	def _optimizer(self):
		"""
		initiate the deep learning optimizer: step, momentum...
		"""
		step_size = 0.015

	def compute_gradients(self, total_steps:int, batch_size:int):
		"""
		compute the gradients of the policy w.r.t. the NN parameters
		Parameter:
			input : current state labeled in normalized (x,y)
		"""
		env = QubitEnv(total_steps)
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

	def predict(self,initial_state:Qubit):
		"""
		evaluate the policy to generate an action_type by
		their probability normalized by softmax function 
		"""
		x,y = tp2xy(initial_state.ThetaPhi[0],initial_state.ThetaPhi[1])
		outputs = array([x,y])
		for w, b in self.params :
			activs = (self.activation_func)(outputs)
			outputs = jnp.dot(w, activs) + b

		proba_action = jax.nn.softmax(outputs)
		self.random_key, subkey = random.split(self.random_key)
		action_type = int(random.choice(subkey,7,p=proba_action))-3
		return action_type
	

