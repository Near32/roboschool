from roboschool.scene_abstract import cpp_household
#from roboschool.scene_stadium import SinglePlayerStadiumScene
from roboschool.gym_robosumo import RoboschoolRoboSumo
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys

class RoboschoolRoboSumoMujocoXML(RoboschoolRoboSumo, RoboschoolMujocoXmlEnv):
    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        '''
        TODO: define arguments for RoboSumo...
        '''
        RoboschoolMujocoXmlEnv.__init__(self, fn, robot_name, action_dim, obs_dim)
        RoboschoolRoboSumo.__init__(self, power)


class RoboschoolSumo(RoboschoolRoboSumoMujocoXML):
    foot_list = ['ant0_front_left_foot', 'ant0_front_right_foot', 'ant0_left_back_foot', 'ant0_right_back_foot',
                    'ant1_front_left_foot', 'ant1_front_right_foot', 'ant1_left_back_foot', 'ant1_right_back_foot']
    def __init__(self):
        RoboschoolRoboSumoMujocoXML.__init__(self, fn="2sumos.xml", robot_name=["ant0","ant1"], action_dim=2*8, obs_dim=2*28, power=2.5)
    def alive_bonus(self, z, pitch):
        '''
`       TODO: define it 
        '''
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground
