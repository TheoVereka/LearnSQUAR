# define here the functions that implement the action of gates on the qubit state
# use these functions inside the RL environment class
from jax.numpy import pi,cos,sin,arccos,arctan2,exp,array,mod,linalg,sum,zeros,mean,equal,greater_equal,less,floor
from jax import random, grad, vmap, jit
import qutip as qt
import matplotlib


class Qubit:
	"""
	Custom class which contains the physics
	Parameters:
		...
	"""

	def __init__(self, theta=0, phi=0):
		"""
		Create a qubit from the given Bloch angles,
		if angles are not fully given, create a random one with
		probability uniformly distributed on the spherical measure.
		"""
		self.ThetaPhi = (theta, phi)

	def randomSphr(self, random_key):
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
		fidelity = (1.0 + cos(self.ThetaPhi[0]))/2
		return fidelity



	def apply(self, action):
		"""
		Parameters: 
			gates: QGate objects, simply lambda functions that rotate (theta, phi)
				   along Z-,Y-,X-,none,X+,Y+,Z+ axis with angle = delta_t
			gate_type: 0~6 integers labeling Z-,Y-,X-,none,X+,Y+,Z+ axis rotation
		
		Return:
			rotated (theta' , phi')
		"""
		self.ThetaPhi = (action)(self.ThetaPhi)


	def render_Bloch_repr(self):
		bSphr = qt.Bloch()
		bSphr.add_vectors(self.coordXYZ())
		bSphr.show()








