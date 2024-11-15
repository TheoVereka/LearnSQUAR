# define here the functions that implement the action of gates on the qubit state
# use these functions inside the RL environment class
from jax.numpy import pi,cos,sin,arccos,arctan2,exp,array,mod,linalg,sum
from jax import random
import qutip as qt
import matplotlib
import numpy.random as nprd


class Qubit:
	"""
	Custom class which contains the physics
	Parameters:
		...
	"""

	def __init__(self, random_key=None, theta=None, phi=None, ):
		"""
		Create a qubit from the given Bloch angles,
		if angles are not fully given, create a random one with
		probability uniformly distributed on the spherical measure.
		"""
		if (theta is None) or (phi is None): 
			random3Dcoords=random.normal(random_key,3)
			phi = pi+arctan2(-random3Dcoords[1],-random3Dcoords[0])
			theta = arccos(random3Dcoords[2]/linalg.norm(random3Dcoords))
		
		self.ThetaPhi = (theta, phi)


	def coordPC2(self):
		"""Coordinates in Hilbert space PC2"""
		theta, phi = self.ThetaPhi
		return array([ exp(0.5j*phi)*cos(theta/2) , exp(-0.5j*phi)*sin(theta/2) ])
	

	def coordXYZ(self):
		"""Coordinates in XYZ Bloch sphere"""
		theta, phi = self.ThetaPhi
		return array([sin(theta)*cos(phi) , sin(theta)*sin(phi) , cos(theta)])

	def compute_fidelity(self):
		"""
		Parameter: none, due to simplicity of problem we only consider target_coord = [1;0]
		Return: fidelity = |<target|psi>|^2 = cos(theta/2)^2
		"""
		theta, phi = self.ThetaPhi
		fidelity = (1. + cos(theta))/2
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
		bSphr.add_vectors(states[0].coordXYZ())

		for i in range(1,T+1):
			states[i].ThetaPhi = (actions[i-1])(states[i-1].ThetaPhi)
			rewards.append( states[i].compute_fidelity() )
			bSphr.add_vectors(states[i].coordXYZ())
	
		cmap = matplotlib.pyplot.get_cmap('inferno', T+1)    # PiYG
		bSphr.vector_color = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(cmap.N)]
		bSphr.vector_width = 2
		bSphr.vector_alpha = [0.33]*(T+1)
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

	def match_gate_type(self, action_type, state=None):
		if isinstance(state,Qubit): theta, phi = state.ThetaPhi
		else: phi = pi/4
		match action_type:
			case -3: gate_func = self.Zn
			case -2: gate_func = self.Yn
			case -1: gate_func = self.Xn
			case 0: gate_func = self.Id
			case 1: gate_func = self.Xp
			case 2: gate_func = self.Yp
			case 3: gate_func = self.Zp
			case _: print("Appling qubit-gate's type error")
		return gate_func
