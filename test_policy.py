from policy import *
myPo = Policy()
myPo._architecture()
initial_states, trajectories_type = myPo.compute_gradients(3,5)
print()
print([state.ThetaPhi for state in initial_states])
print()
print(trajectories_type)
