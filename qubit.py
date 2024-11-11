# define here the functions that implement the action of gates on the qubit state
# use these functions inside the RL environment class


class QubitEnv():
    """
    Custom class which contains the physics
    Parameters:
       ...
    """

    def __init__(self):
    	pass


	def compute_fidelity(self):
		pass



	def apply_gate(self):
		pass


	def render_Bloch_repr(self)
		pass


	def evaluate_trajectory(self, initial_state, trajectory):
		"""
		Parameters:
			Takes trajectory of unitaries and applies them on the initial_state

		Returns:
			the state and the reward at every step along the trajectory
		"""
		pass



