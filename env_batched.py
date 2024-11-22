# use the template below to define the environment, see r.g. https://github.com/openai/gym/blob/master/gym/envs/classic_control/acrobot.py
from qubit_batched import *
import jax
import jax.numpy as jnp


def step(tp,a):
    """
    redefine gates_type: [0:+X, 1:+Y, 2:-X, 3:-Y, 4:+Z, 5:-Z, 7:Id]

    match a:
        case 6: tp = array([tp[0],tp[1]])
        case 4: tp = array([tp[0],mod(tp[1]+delta,2*pi)])
        case 5: tp = array([tp[0],mod(tp[1]-delta,2*pi)])
        case _: 
            match mod(a-floor(tp[1]/pi*2),4):
                case 3: tp = array([ arccos(cos(tp[0])*cos(delta)+sin(tp[0])*cos(tp[1])*sin(delta)) , pi+arctan2(-sin(tp[0])*sin(tp[1]),cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) ])
                case 2: tp = array([ arccos(cos(tp[0])*cos(delta)-sin(tp[0])*sin(tp[1])*sin(delta)) , pi+arctan2(-cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) ])
                case 0: tp = array([ arccos(cos(tp[0])*cos(delta)+sin(tp[0])*sin(tp[1])*sin(delta)) , pi+arctan2(cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) ])
                case 1: tp = array([ arccos(cos(tp[0])*cos(delta)-sin(tp[0])*cos(tp[1])*sin(delta)) , pi+arctan2(-sin(tp[0])*sin(tp[1]),-cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) ])
                case _: return "ERROR!"
    """
    delta = 4*pi/60
    a_rot = mod(a-jnp.floor(tp[1]/pi*2),4)
    tp = array([
        jnp.greater_equal(a,4)*tp[0]+
        jnp.less(a,4)*(
            equal(a_rot,3)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*cos(tp[1])*sin(delta)) +
            equal(a_rot,2)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*sin(tp[1])*sin(delta)) +
            equal(a_rot,0)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*sin(tp[1])*sin(delta)) +
            equal(a_rot,1)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*cos(tp[1])*sin(delta)) )
        ,
        equal(a,6)*tp[1]+
        equal(a,4)*mod(tp[1]+delta,2*pi)+
        equal(a,5)*mod(tp[1]-delta,2*pi)+
        jnp.less(a,4)*(
            equal(a_rot,3)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) )  +
            equal(a_rot,2)*( pi+arctan2(-cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) ) +
            equal(a_rot,0)*( pi+arctan2(cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) )  +
            equal(a_rot,1)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),-cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) ) )
    ])
    return tp , (1.0+cos(tp[0]))/2  # , True

class QubitEnv:
    """
    Gym style environment for RL
    Parameters:
        n_time_steps:   int
                        Total number of time steps within each episode
    """


    def __init__(self, n_time_steps, batch_size, random_key):
        """
        Initialize the qubit randomly as the initial_state
        Zn = lambda tp: array(tp[0], tp[1] - delta)
        Yn = lambda tp: array( arccos(cos(tp[0])*cos(delta)+sin(tp[0])*cos(tp[1])*sin(delta)) , pi+arctan2(-sin(tp[0])*sin(tp[1]),cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) )
        Xn = lambda tp: array( arccos(cos(tp[0])*cos(delta)-sin(tp[0])*sin(tp[1])*sin(delta)) , pi+arctan2(-cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) )
        Id = lambda tp: array( tp[0], tp[1] )
        Xp = lambda tp: array( arccos(cos(tp[0])*cos(delta)+sin(tp[0])*sin(tp[1])*sin(delta)) , pi+arctan2(cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) )
        Yp = lambda tp: array( arccos(cos(tp[0])*cos(delta)-sin(tp[0])*cos(tp[1])*sin(delta)) , pi+arctan2(-sin(tp[0])*sin(tp[1]),-cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) )
        Zp = lambda tp: array( tp[0] , tp[1] + delta )
        self.gate = array(tp[0], tp[1] - delta) ###[TODO]: For later quadrant's use
        """
        self.delta = 4*pi/n_time_steps

        def generateThetaPhi(key): 
            state = Qubit()
            state.randomSphr(key)
            return array([state.ThetaPhi[0], state.ThetaPhi[1]])
        self.batch_ThetaPhi = ( vmap(generateThetaPhi,in_axes=0,out_axes=0) )( random.split(random_key, batch_size) )


                

    def step_monstreous(self, tp, a):
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
        delta = self.delta
        ThetaPhi = array([

            (equal(a,0)*tp[0]
            +equal(a,1)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*cos(tp[1])*sin(delta))
            +equal(a,2)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*sin(tp[1])*sin(delta))
            +equal(a,3)*tp[0]
            +equal(a,4)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*sin(tp[1])*sin(delta))
            +equal(a,5)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*cos(tp[1])*sin(delta))
            +equal(a,6)*tp[0])

            *greater_equal(tp[1],0.0)*less(tp[1],pi/2)+

            (equal(a,0)*tp[0]
            +equal(a,1)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*sin(tp[1])*sin(delta))
            +equal(a,2)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*cos(tp[1])*sin(delta))
            +equal(a,3)*tp[0]
            +equal(a,4)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*cos(tp[1])*sin(delta))
            +equal(a,5)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*sin(tp[1])*sin(delta))
            +equal(a,6)*tp[0])

            *greater_equal(tp[1],pi/2)*less(tp[1],pi)+

            (equal(a,0)*tp[0]
            +equal(a,1)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*cos(tp[1])*sin(delta))
            +equal(a,2)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*sin(tp[1])*sin(delta))
            +equal(a,3)*tp[0]
            +equal(a,4)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*sin(tp[1])*sin(delta))
            +equal(a,5)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*cos(tp[1])*sin(delta))
            +equal(a,6)*tp[0])

            *greater_equal(tp[1],pi)*less(tp[1],3*pi/2)+

            (equal(a,0)*tp[0]
            +equal(a,1)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*sin(tp[1])*sin(delta))
            +equal(a,2)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*cos(tp[1])*sin(delta))
            +equal(a,3)*tp[0]
            +equal(a,4)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*cos(tp[1])*sin(delta))
            +equal(a,5)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*sin(tp[1])*sin(delta))
            +equal(a,6)*tp[0])

            *greater_equal(tp[1],3*pi/2)*less(tp[1],pi*2)

            ,

            (equal(a,0)*mod(tp[1]-delta,2*pi)
            +equal(a,1)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) )
            +equal(a,2)*( pi+arctan2(-cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) )
            +equal(a,3)*tp[1]
            +equal(a,4)*( pi+arctan2(cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) ) 
            +equal(a,5)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),-cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) )
            +equal(a,6)*mod(tp[1]+delta,2*pi))

            *greater_equal(tp[1],0.0)*less(tp[1],pi/2)+

            (equal(a,0)*mod(tp[1]-delta,2*pi)
            +equal(a,1)*( pi+arctan2(cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) ) 
            +equal(a,2)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) )
            +equal(a,3)*tp[1]
            +equal(a,4)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),-cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) )
            +equal(a,5)*( pi+arctan2(-cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) )
            +equal(a,6)*mod(tp[1]+delta,2*pi))

            *greater_equal(tp[1],pi/2)*less(tp[1],pi)+

            (equal(a,0)*mod(tp[1]-delta,2*pi)
            +equal(a,1)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),-cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) )
            +equal(a,2)*( pi+arctan2(cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) ) 
            +equal(a,3)*tp[1]
            +equal(a,4)*( pi+arctan2(-cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) )
            +equal(a,5)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) )
            +equal(a,6)*mod(tp[1]+delta,2*pi))

            *greater_equal(tp[1],pi)*less(tp[1],3*pi/2)+

            (equal(a,0)*mod(tp[1]-delta,2*pi)
            +equal(a,1)*( pi+arctan2(-cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) )
            +equal(a,2)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),-cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) )
            +equal(a,3)*tp[1]
            +equal(a,4)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) )
            +equal(a,5)*( pi+arctan2(cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) ) 
            +equal(a,6)*mod(tp[1]+delta,2*pi))

            *greater_equal(tp[1],3*pi/2)*less(tp[1],pi*2)

        ])
        reward = (1.0 + cos(ThetaPhi[0]))/2
        return ThetaPhi, reward #, True

    def step_normalizedTP(self, tp, a):
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
        delta = self.delta
        ThetaPhi = array([
            (equal(a,0)*tp[0]
            +equal(a,1)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*cos(tp[1])*sin(delta))
            +equal(a,2)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*sin(tp[1])*sin(delta))
            +equal(a,3)*tp[0]
            +equal(a,4)*arccos(cos(tp[0])*cos(delta)+sin(tp[0])*sin(tp[1])*sin(delta))
            +equal(a,5)*arccos(cos(tp[0])*cos(delta)-sin(tp[0])*cos(tp[1])*sin(delta))
            +equal(a,6)*tp[0])
            ,
            (equal(a,0)*mod(tp[1]-delta,2*pi)
            +equal(a,1)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) )
            +equal(a,2)*( pi+arctan2(-cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) )
            +equal(a,3)*tp[1]
            +equal(a,4)*( pi+arctan2(cos(tp[0])*sin(delta)-sin(tp[0])*sin(tp[1])*cos(delta),-sin(tp[0])*cos(tp[1])) ) 
            +equal(a,5)*( pi+arctan2(-sin(tp[0])*sin(tp[1]),-cos(tp[0])*sin(delta)-sin(tp[0])*cos(tp[1])*cos(delta)) )
            +equal(a,6)*mod(tp[1]+delta,2*pi))
        ])
        reward = (1.0 + cos(ThetaPhi[0]))/2
        return ThetaPhi, reward #, True













