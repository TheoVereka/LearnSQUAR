# define here the policy gradient algorithm with the training loop
from policy import *


class Policy_Gradient():

	def __init__(self):
		"""
		init NN
		"""
		pass


	def train_policy(self,n_epoch):
		"""
		Loop over epochs
			collect_traj(policy.params)
			update_params( to get grad(loss_functional)(params) ...)

		"""
		for epoch in len(self.policy.batch_size):
			True
		pass


	def evaluate_policy(self):
		"""
		Evaluate performance of trained policy: always take most proba action
		"""
		pass


	def plot_returns(self):
		"""
		Plot the returns as a function of episode number: final reward/fidelity
		"""
		pass
