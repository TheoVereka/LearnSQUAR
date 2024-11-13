# define here the deep neural network representation of the policy
from env import *

#%%
# a parametrization which can avoid the singularity in spherical angles' poles
def tp2xy(theta,phi):
    x = theta/pi + 1/2 - 1/pi*arccos((4*phi/pi-1)*sin(theta))
    y = theta/pi - 1/2 + 1/pi*arccos((4*phi/pi-1)*sin(theta))
    return (x,y)

def xy2tp(x,y):
    theta = pi/2*(x+y)
    phi = pi/4*(1+sin(pi/2*(x-y))/sin(pi/2*(x+y)))
    return (theta,phi)


class Policy():

	def __init__(self):
		pass


	def _architecture(self):
		"""
		contains the NN architecture
		"""
		pass

	def _optimizer(self):
		"""
		initiate the deep learning optimizer with close physical guess
		"""
		pass

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

		for i in range(batch_size):
			env.reset()
			trajectories_type.append([])
			initial_states.append(env.state)
			total_reward = 0
			# grad_batch[eta] = 0

			for t in range(total_steps):
				action_type = self.predict(env.state)
				trajectories_type[i].append(action_type)
				# grad_batch[eta] += autodifferation_of_log_pi(action_type|state)
				env.state.apply(env.gates,action_type)
				total_reward += env.state.compute_fidelity()
			#  grandient[eta]+= total_reward * grad_batch[eta]

		# gradient[eta] /= batch_size

	def predict(self,initial_state:Qubit):
		"""
		evaluate the policy to generate an action_type by
		their probability normalized by softmax function 
		"""
		x,y = tp2xy(initial_state.ThetaPhi[0],initial_state.ThetaPhi[1])

		#input x,y --eta--> output values --softmax--> proba
		
		proba_action = [0.10, 0.02, 0.74, 0.01, 0.03, 0.06, 0.04]
		action_type = int(np.random.choice(7,1,p=proba_action))-3
		return action_type
	



