
#%%
from qubit import *
import jax.numpy as jnp
import numpy.random as rd
import matplotlib.pyplot as plt
pi = jnp.pi
T = 60 

# test two classes in qubit.py
qb = Qubit(0.2, 0.3)
gateT = QGates(pi*4/T)


#%%
# test compute_fidelity
print(qb.compute_fidelity())


#%%
# test manually set qubit to |-x> and render it
qb.ThetaPhi = (pi/2, pi)
print(qb.ThetaPhi)
qb.render_Bloch_repr()
for i in range(1): qb.ThetaPhi = gateT.Yp(qb.ThetaPhi)
qb.render_Bloch_repr()
for i in range(20): qb.apply(gateT,-2)
qb.render_Bloch_repr()

# %%
# test evaluate_trajectory with applying gates of Y-
actions_type = rd.randint(7,size=T)-3
actions = [gateT.match_gate_type(actionType) for actionType in actions_type]
plt.plot(actions_type)
states, rewards = qb.evaluate_trajectory(Qubit(pi-0.1, pi/2),actions)
plt.plot(rewards)

# %%
# test random initialization

rndQB = Qubit()
rndQB.render_Bloch_repr()

# %%
