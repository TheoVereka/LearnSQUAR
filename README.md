# LearnSQUAR 
Miniproject of reinforcement rearning for a Single Qubit's Unitary Active Reset  
JAX 0.4.35, Python 3.10.11, VSCode 1.95.2, CPU i9-13900H  
  
Goal:  
Train a RL agent to reset a qubit under unitary gate evolution  
  
Definition:  
reset: arbitrary state to |down>:=(1;0)  
qubit: state in Hilbert space PC2, parametrized on a Bloch sphere  
unitary gate evolution: generators are Pauli matrices or 0 times i*δt/2  
episodic RL evironment: tranched time stamp, on which define states, between states define actions and corresponding reward  
RL agent: input current state -> network with inner to-train parameters -> output probality distribution of actions (RL policy)  
  
TODO:  
- Parametrization with no degeneracy and discontinuity
- Autobatch vmap
- think about how to train the policy (varying nothing? batch_size? even T_steps in training?)
 
