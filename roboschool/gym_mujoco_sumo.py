from roboschool.scene_abstract import cpp_household
#from roboschool.scene_stadium import SinglePlayerStadiumScene
from roboschool.gym_forward_walker import RoboschoolForwardWalker
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys

class RoboschoolRoboSumoMujocoXML(RoboschoolRoboSumo, RoboschoolMujocoXmlEnv):
    def __init__(self, fn, robot_name, action_dim, obs_dim):
        '''
        TODO: define arguments for RoboSumo...
        '''
        RoboschoolMujocoXmlEnv.__init__(self, fn, robot_name, action_dim, obs_dim)
        RoboschoolRoboSumo.__init__(self)


class RoboschoolAnt(RoboschoolRoboSumoMujocoXML):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']
    def __init__(self):
        RoboschoolRoboSumoMujocoXML.__init__(self, "ant.xml", "torso", action_dim=8, obs_dim=28, power=2.5)
    def alive_bonus(self, z, pitch):
        '''
`       TODO: define it 
        '''
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground
