#from OpenGL import GL 
import roboschool
import gym
import numpy as np 
import time 

env = gym.make('RoboschoolSumo-v0')
env.reset()
while True:
    # The first sumo does not move, while the second moves randomly:
    action = env.action_space.sample()
    action[0][0:8] = 0.0
    action[1][0:8] = 0.0
    #action = (action[1], action[1])
    state, reward, done, info = env.step(action)
    
    pos = np.array([ [state[0][0],state[1][0] ],
                        [state[0][1],state[1][1] ],
                        [state[0][2],state[1][2] ]])
    q = np.array([ [state[0][3],state[1][3] ],
                        [state[0][4],state[1][4] ],
                        [state[-1][5],state[1][5] ],
                        [state[0][6],state[1][6] ]])
    print(q)
    print('--'*20)
    #env.render()
    time.sleep(1)
    
    if done :
        env.reset()
        break
        #print('environment resetted...')
