# define here the functions that implement the action of gates on the qubit state
# use these functions inside the RL environment class
from jax.numpy import pi,cos,sin,arccos,arctan2,exp,array
import qutip as qt
import numpy as np


class Qubit:
	"""
	Custom class which contains the physics
	Parameters:
		...
	"""

	def __init__(self, theta, phi ):
		self.ThetaPhi = (theta, phi)

	def coordPC2(self):
		"""Coordinates in Hilbert space PC2"""
		theta, phi = self.ThetaPhi
		return array([ exp(0.5j*phi)*cos(theta/2) , exp(-0.5j*phi)*sin(theta/2) ])
	

	def coordXYZ(self):
		"""Coordinates in XYZ Bloch sphere"""
		theta, phi = self.ThetaPhi
		return array([np.sin(theta)*np.cos(phi) , np.sin(theta)*np.sin(phi) , np.cos(theta)])

	def compute_fidelity(self):
		"""
		Parameter: none, due to simplicity of problem we only consider target_coord = [1;0]
		Return: fidelity = |<target|psi>|^2 = cos(theta/2)^2
		"""
		theta, phi = self.ThetaPhi
		fidelity = (1. + np.cos(theta))/2
		return fidelity



	def apply(self, gates, gate_type):
		"""
		Parameters: 
			gates: QGate objects, simply lambda functions that rotate (theta, phi)
				   along Z-,Y-,X-,none,X+,Y+,Z+ axis with angle = delta_t
			gate_type: -3~+3 integers labeling Z-,Y-,X-,none,X+,Y+,Z+ axis rotation
		
		Return:
			rotated (theta' , phi')
		"""
		self.ThetaPhi = (gates.match_gate_type(gate_type))(self.ThetaPhi)


	def render_Bloch_repr(self):
		bSphr = qt.Bloch()
		bSphr.add_vectors(self.coordXYZ())
		bSphr.show()


	def evaluate_trajectory(self, initial_state, trajectory):
		"""
		Parameters:
			Takes trajectory of unitaries and applies them on the initial_state
			initial_state: Qubit object
			trajectory: the actions sequence of gate_func (rotation functions)

		Returns:
			the state and the reward at every step along the trajectory
		"""
		T = len(trajectory)
		states = [initial_state]*(T+1)
		actions = trajectory
		rewards = [initial_state.compute_fidelity()]
		bSphr = qt.Bloch()
		bSphr.add_points(states[0].coordXYZ())

		for i in range(1,T+1):
			states[i].ThetaPhi = (actions[i-1])(states[i-1].ThetaPhi)
			rewards.append( states[i].compute_fidelity() )
			bSphr.add_vectors(states[i].coordXYZ())

		bSphr.show()

		return states, rewards




class QGates:
	"""
	Class of unitary transformation for qubit's PC2 Hilbert space
	represented as Bloch sphere's theta and phi angles' rotation

	Currently only include seven rotations with [delta_t] as variable
	"""

	def __init__(self, delta):
		"""
		delta = delta_t = 4 pi / T
		"""
		self.Zn = lambda tp: (tp[0], tp[1] - delta)
		self.Yn = lambda tp: ( arccos(cos(tp[0])*cos(delta)+sin(tp[0])*cos(tp[1])*sin(delta)) , pi+arctan2(-sin(tp[0])*sin(tp[1]),cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) )
		self.Xn = lambda tp: ( arccos(cos(tp[0])*cos(delta)-sin(tp[0])*sin(tp[1])*sin(delta)) , pi+arctan2(-cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) )
		self.Id = lambda tp: ( tp[0], tp[1] )
		self.Xp = lambda tp: ( arccos(cos(tp[0])*cos(delta)+sin(tp[0])*sin(tp[1])*sin(delta)) , pi+arctan2(cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) )
		self.Yp = lambda tp: ( arccos(cos(tp[0])*cos(delta)-sin(tp[0])*cos(tp[1])*sin(delta)) , pi+arctan2(-sin(tp[0])*sin(tp[1]),-cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) )
		self.Zp = lambda tp: ( tp[0] , tp[1] + delta )

	def match_gate_type(self, gate_type):
		match gate_type:
			case -3: gate_func = self.Zn
			case -2: gate_func = self.Yn
			case -1: gate_func = self.Xn
			case 0: gate_func = self.Id
			case 1: gate_func = self.Xp
			case 2: gate_func = self.Yp
			case 3: gate_func = self.Zp
			case _: print("Appling qubit-gate's type error")
		return gate_func
