# use the template below to define the environment, see r.g. https://gym.openai.com/envs/#classic_control


class QubitEnv():
    """
    Gym style environment for RL
    Parameters:
        n_time_steps:   int
                        Total number of time steps within each episode
    """

    def __init__(self, n_time_steps):

    	# initiate the environment
        pass

    def step(self, action):
        """
        Interface between environment and agent. Performs one step in the environemnt.
        Parameters:
            action: int
                    the index of the respective action in the action array
        Returns:
            output: ( array, float, bool)
                    information provided by the environment about its current state:
                    (state, reward, done)
        """

       	pass

        return self.state, reward, done

    def reset(self):
        """
        Resets the environment to its initial values.
        Returns:
            state:  array
                    the initial state of the environment
        """
        pass

        return self.state

    def render(self):
    	"""
    	Plots the state as an arrow on the Bloch sphere. 

    	"""
        pass


