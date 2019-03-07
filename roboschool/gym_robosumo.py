from roboschool.scene_abstract import cpp_household
from roboschool.scene_tatami import SinglePlayerTatamiScene
from roboschool.multiplayer import SharedMemoryClientEnv
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys


class RoboschoolRoboSumo(SharedMemoryClientEnv):
    COST_COEFS = {
        'ctrl': 1e-1,
        # 'pain': 1e-4,
        # 'attack': 1e-1,
    }
    tatami_radius = 3.0
    tatami_height = 0.25

    def __init__(self, power):
        self.power = power
        self.camera_x = 0
        '''
        TODO: add reward shaping for the sumos to stay in the middle of the tatami.
        '''
        self.sumo_target_x = 0
        self.sumo_target_y = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.camera_x = 0
        self.camera_y = 4.3
        self.camera_z = 45.0
        self.camera_follow = 0

    def create_single_player_scene(self):
        return SinglePlayerTatamiScene(gravity=9.8, timestep=0.0165/4, frame_skip=4)

    def robot_specific_reset(self):
        for j in self.ordered_joints:
            j.reset_current_position(self.np_random.uniform( low=-0.1, high=0.1 ), 0)
        self.feet = {}
        for r in self.cpp_robots:
            self.feet[r] = []
            for f in self.foot_list :
                if f in self.parts[r]:
                   self.feet[r].append( self.parts[r][f] )
        self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
        self.scene.actor_introduce(self)
        
    def move_robot(self, init_x, init_y, init_z):
        '''
        TODO: used by multiplayer stadium to move the agent around the tatami
        '''
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robots[-1].query_position()
        pose = self.cpp_robots[-1].root_part.pose()
        pose.move_xyz(init_x, init_y, init_z)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robots[-1].set_pose(pose)
        self.start_pos_x, self.start_pos_y, self.start_pos_z = init_x, init_y, init_z

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        for n,j in enumerate(self.ordered_joints):
            j.set_motor_torque( self.power*j.power_coef*float(np.clip(a[n], -1, +1)) )

    def _compute_after_step(self, robot_idx, action):
        self.posafter = self.robot_bodies[robot_idx].pose().xyz()
        # Control cost
        reward = - self.COST_COEFS['ctrl'] * np.square(action).sum()
        return reward

    def _compute_alive(self):
        self.contact_with_floor = {}
        pr_contacts = {} 
        for cppr in self.cpp_robots:
            pr_contacts[cppr] = []
            for part_name in self.parts[cppr]:
                contact_names = [ x.name for x in self.parts[cppr][part_name].contact_list()]
                pr_contacts[cppr] += contact_names 
            self.contact_with_floor[cppr] = 1.0 if not('floor' in pr_contacts[cppr]) else 0.0
        return self.contact_with_floor.values()

    electricity_cost     = -2.0    # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost    = -0.1    # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost  = -1.0    # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
    joints_at_limit_cost = -0.2    # discourage stuck joints

    def calc_state(self):
        self.state = {}
        
        # Joints:
        j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        pr_joints = {}
        for cppr in self.cpp_robots:
            pr_joints[cppr] = np.concatenate([ self.jdict[cppr][joint_name].current_relative_position() for joint_name in self.jdict[cppr] ]).flatten()
        # nbr_robots x nbr_joints(==2 * nbr_legs == 8) * 2 == 2 x 16
        self.joint_positions = j[0::1]
        self.joint_speeds = j[1::2]
        self.pr_joints_at_limit = [ np.count_nonzero(np.abs(prj[0::2]) > 0.99) for prj in np.split(j, 2, axis=0) ]
        self.state["joints"] = pr_joints
        
        # Poses:
        pr_poses = {}
        for cppr,r in zip(self.cpp_robots,self.robot_bodies):
            pr_poses[cppr] = np.concatenate( [r.pose().xyz(), r.pose().quatertion()]).flatten()
        # nbr_robots x 7
        self.state["poses"] = pr_poses
        
        bodies_pose = [b.pose() for b in self.robot_bodies]
        parts_xyz = [ np.array( [p.pose().xyz() for p in self.parts[cppr].values()] ).flatten() for cppr in self.cpp_robots]
        self.bodies_xyz = [ (p_xyz[0::3].mean(), p_xyz[1::3].mean(), b_pose.xyz()[2]) for p_xyz, b_pose in zip(parts_xyz,bodies_pose) ]  # torso z is more informative than mean z
        self.body_xyz = self.bodies_xyz[0]
        self.bodies_rpy = [ b_pose.rpy() for b_pose in bodies_pose]

        # Contacts: 
        pr_contacts = {} 
        for cppr in self.cpp_robots:
            pr_contacts[cppr] = np.array( [ len(self.parts[cppr][part_name].contact_list() ) for part_name in self.parts[cppr] ] ).flatten()
        # nbr_robots x nbr_parts (==8) * 1
        self.state["contacts"] = pr_contacts
        
        # Observations:
        obs = [ [] for cppr in self.cpp_robots]
        for idxr, cppr in enumerate(self.cpp_robots):
            obs[idxr].append(self.state["poses"][cppr]) 
            # pose: x,y,z, qx,qy,qz,qw : 7
            obs[idxr].append( self.state["joints"][cppr])
            # joints: nbr_joints(==2 * nbr_legs == 8) * 2 ==  16
            obs[idxr].append( self.state["contacts"][cppr])
            # contacts: nbr_parts*1 == 20 
            for cppo in self.cpp_robots:
                if cppr != cppo :
                    obs[idxr].append(self.state["poses"][cppo]) 
                    # opponent pose: x,y,z, qx,qy,qz,qw : 7
            obs[idxr] = np.concatenate(obs[idxr]).flatten()
            # 7 + 16 + 20 + 7*nbr_opponent(==1) == 50
        
        self.state4HUD = np.concatenate(obs).flatten()
        per_robot_obs = obs

        return per_robot_obs
    
    def _step(self, a):
        # :param a: assumes either a list of actions for each sumo 
        # or a concatenation of both actions into one array, 
        # with sumo0's actions followed by sumo1's actions.
        a = np.concatenate([*a], axis=0).flatten()

        # the multiplayer support for this environment is embedded into the step/reset mechanisms...
        assert(not self.scene.multiplayer)  
        
        self.apply_action(a)
        self.scene.global_step()

        dones = [False for _ in range(len(self.cpp_robots))]
        rewards = [0. for _ in range(len(self.cpp_robots))]
        infos = [{} for _ in range(len(self.cpp_robots))]

        for i in range(len(self.cpp_robots)):
            infos[i]['ctrl_reward'] = self._compute_after_step(i, a[i])
        
        per_robot_obs = self.calc_state()

        alives = self._compute_alive()
        done = False
        draw = sum(alives) <= 0.0 or sum(alives) >= 2.0
        for alive in alives: 
            if not alive: 
                done = True
                break

        for key in self.state:
            for cppr in self.state[key]:
                for el in self.state[key][cppr]:
                    if not np.isfinite(el).all():
                        done = True
                        break 

        # Costs:
        pr_electricity_cost  = [ self.stall_torque_cost * float(np.square(pr_a).mean()) + self.electricity_cost  * float(np.abs(pr_a*pr_joint_speeds).mean()) for pr_a, pr_joint_speeds in zip( np.split(a, 2, axis=0), np.split(self.joint_speeds, 2, axis=0))]   # let's assume we have DC motor with controller, and reverse current braking
        pr_joints_at_limit_cost = [ float(self.joints_at_limit_cost) * jtl for jtl in self.pr_joints_at_limit ]

        self.dense_rewards = [] 
        for alive, el_cost, limit_cost in zip(alives,pr_electricity_cost,pr_joints_at_limit_cost):
            self.dense_rewards.append( [alive, el_cost, limit_cost] )

        self.frame  += 1
        if (done and not self.done) or self.frame==self.spec.timestep_limit:
            self.done = True 
            self.episode_over(self.frame)
        else :
            self.done = done

        self.rewards = [0.0 for _ in self.cpp_robots]
        if self.done:
            if draw : 
                self.rewards = [-1000 for _ in self.cpp_robots]
            else :
                for idx, alive in enumerate(alives):
                    pr_reward = 2000.0 if alive else -2000.0
                    self.rewards[idx] = pr_reward
                
        for i in range(len(self.cpp_robots)):
            infos[i]['dense_rewards'] = self.dense_rewards[i]
        
        self.HUD(self.state4HUD, a, done)

        return per_robot_obs, self.rewards, self.done, infos

    def episode_over(self, frames):
        self.done = True 

    def camera_adjust(self):
        #self.camera_dramatic()
        self.camera_simple_follow()

    def camera_simple_follow(self):
        x, y, z = self.body_xyz
        self.camera_x = 0.98*self.camera_x + (1-0.98)*x
        self.camera.move_and_look_at(self.camera_x, y-2.0, 1.4, x, y, 1.0)

    def camera_dramatic(self):
        bodies_pose = [r.pose() for r in self.robot_bodies]
        bodies_speed = [r.speed() for r in self.robot_bodies]
        x, y, z = np.mean( [p.xyz() for p in pose], axis=0 )
        if 1:
            camx, camy, camz = speed[0], speed[1], 2.2
        else:
            camx, camy, camz = self.walk_target_x - x, self.walk_target_y - y, 2.2

        n = np.linalg.norm([camx, camy])
        if n > 2.0 and self.frame > 50:
            self.camera_follow = 1
        if n < 0.5:
            self.camera_follow = 0
        if self.camera_follow:
            camx /= 0.1 + n
            camx *= 2.2
            camy /= 0.1 + n
            camy *= 2.8
            if self.frame < 1000:
                camx *= -1
                camy *= -1
            camx += x
            camy += y
            camz  = 1.8
        else:
            camx = x
            camy = y + 4.3
            camz = 2.2
        #print("%05i" % self.frame, self.camera_follow, camy)
        smoothness = 0.97
        self.camera_x = smoothness*self.camera_x + (1-smoothness)*camx
        self.camera_y = smoothness*self.camera_y + (1-smoothness)*camy
        self.camera_z = smoothness*self.camera_z + (1-smoothness)*camz
        self.camera.move_and_look_at(self.camera_x, self.camera_y, self.camera_z, x, y, 0.6)

