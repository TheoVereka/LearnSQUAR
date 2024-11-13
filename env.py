# use the template below to define the environment, see r.g. https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py
from qubit import *



class QubitEnv():
    """
    Gym style environment for RL
    Parameters:
        n_time_steps:   int
                        Total number of time steps within each episode
    """

    def __init__(self, n_time_steps):
        """
        Initialize the qubit randomly as the initial_state
        """
        self.state = Qubit()
        self.gates = QGates(4*pi/n_time_steps)
        theta, phi = self.state.ThetaPhi
        phi = np.mod(phi,pi/2)
        self.state.ThetaPhi = (theta,phi)


    def step(self, action_type):
        """
        Interface between environment and agent. Performs one step in the environemnt.
        Parameters:
            action: int
                    the index of the respective action in the action array
        Returns:
            output: ( array, float)
                    information provided by the environment about its current state:
                    (state, reward)
        """
        self.state.apply(self.gates, action_type)
        reward = self.state.compute_fidelity()

        return self.state, reward

    def reset(self):
        """
        Resets the environment to its initial values.
        Returns:
            state:  array
                    the initial state of the environment
        """
        self.state = Qubit()
        theta, phi = self.state.ThetaPhi
        phi = np.mod(phi,pi/2)
        self.qb.ThetaPhi = (theta,phi)

        return self.state

    def render(self):
        """
        Plots the state as an arrow on the Bloch sphere. 

        """
        self.state.render_Bloch_repr()

    def act2traj(self, action_types):
        trajectory = [self.gates.match_gate_type(action_type) for action_type in action_types]
        return trajectory

