from roboschool.scene_abstract import cpp_household
from roboschool.scene_tatami import SinglePlayerTatamiScene
from roboschool.multiplayer import SharedMemoryClientEnv
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys
import math 

class RoboschoolRoboSumo(SharedMemoryClientEnv):
    REWARDCOST_COEFS = {
        'ctrl': 1e-2,
        # 'pain': 1e-4,
        'attack': 1e+1,
    }
    tatami_radius = 3.0
    tatami_height = 0.25

    def __init__(self, power, use_reward_shaping=False):
        self.power = power
        self.use_reward_shaping = use_reward_shaping
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

        #Rotate the second sumo:
        [ cppr.query_position() for cppr in self.cpp_robots]
        pose = self.cpp_robots[1].root_part.pose()
        pose.rotate_z(np.pi) 
        self.cpp_robots[1].set_pose(pose)
        [ cppr.query_position() for cppr in self.cpp_robots]
        
        #Initialize init_poses:
        self.init_poses = [cppr.root_part.pose() for cppr in self.cpp_robots]
        self.init_matrices = [ self._from_pose_to_matrix(pose) for pose in self.init_poses]
        self.init_transforms = [ self._from_matrix_to_transform(matrix) for matrix in self.init_matrices]
        self.inv_init_transforms = [ np.linalg.inv( tr ) for tr in self.init_transforms]

    def move_robot(self, init_x, init_y, init_z):
        '''
        TODO: used by multiplayer stadium to move the agent around the tatami
        '''
        raise NotImplemented 

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        for n,j in enumerate(self.ordered_joints):
            j.set_motor_torque( self.power*j.power_coef*float(np.clip(a[n], -1, +1)) )

    def _compute_control_cost(self,action):
        return -self.REWARDCOST_COEFS['ctrl'] * np.square(action).sum()

    def _compute_attack_reward(self, idx1):
        bodies_speed = [ np.array( b.speed() ).flatten() for b in self.robot_bodies]
        bodies_xyz = [ np.array(b.pose().xyz() ).flatten() for b in self.robot_bodies]
        xyz1 = bodies_xyz[idx1]
        speed1 = bodies_speed[idx1]
        pr_reward = []
        for idx2, xyz2 in enumerate(bodies_xyz):
            if idx1 == idx2:
                continue
            dir_1to2 = (xyz2-xyz1)
            norm = np.linalg.norm(dir_1to2)
            dir_1to2 /= (1e-3+ norm)
            pr_reward.append(self.REWARDCOST_COEFS['attack'] * np.dot(dir_1to2, speed1).sum() ) 
        return sum(pr_reward)

    def _compute_alive(self):
        self.contact_with_floor = {}
        self.contact_with_tatami = {}
        pr_contacts = {} 
        pr_contacts_above_foot = {}
        
        for cppr in self.cpp_robots:
            pr_contacts[cppr] = []
            pr_contacts_above_foot[cppr] = []

            for part_name in self.parts[cppr]:
                contact_names = [ x.name for x in self.parts[cppr][part_name].contact_list()]
                pr_contacts[cppr] += contact_names
                if "foot" in part_name:
                    continue 
                else :
                    pr_contacts_above_foot[cppr] += contact_names

            self.contact_with_floor[cppr] = 1.0 if not('floor' in pr_contacts[cppr]) else 0.0
            self.contact_with_tatami[cppr] = 1.0 if not('tatami' in pr_contacts_above_foot[cppr]) else 0.0 
        
        alives = [ cf*ct for cf,ct in zip(self.contact_with_floor.values(), self.contact_with_tatami.values() )]
        return alives

    electricity_cost     = -1.0    # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost    = -0.1    # cost for running electric current through a motor even at zero rotational speed, small
    joints_at_limit_cost = -0.2    # discourage stuck joints

    def _from_quaternion_to_matrix(self, qx, qy, qz, qw):
        R = np.matrix( [[1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz + 2*qy*qw],
                        [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
                        [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy] ])
        return R    

    def _from_matrix_to_quaternion(self, R):
        tr = R[0,0] + R[1,1] + R[2,2]
        if tr > 0: 
          S = math.sqrt(tr+1.0) * 2 # S=4*qw
          qw = 0.25 * S
          qx = (R[2,1] - R[1,2]) / S;
          qy = (R[0,2] - R[2,0]) / S; 
          qz = (R[1,0] - R[0,1]) / S; 
        elif ((R[0,0] > R[1,1]) and (R[0,0] > R[2,2])): 
          S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2 # S=4*qx 
          qw = (R[2,1] - R[1,2]) / S;
          qx = 0.25 * S;
          qy = (R[0,1] + R[1,0]) / S; 
          qz = (R[0,2] + R[2,0]) / S; 
        elif (R[1,1] > R[2,2]):
          S = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2 # S=4*qy
          qw = (R[0,2] - R[2,0]) / S;
          qx = (R[0,1] + R[1,0]) / S; 
          qy = 0.25 * S;
          qz = (R[1,2] + R[2,1]) / S; 
        else: 
          S = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2 # S=4*qz
          qw = (R[1,0] - R[0,1]) / S;
          qx = (R[0,2] + R[2,0]) / S;
          qy = (R[1,2] + R[2,1]) / S;
          qz = 0.25 * S;
        
        return [qx, qy, qz, qw]

    def _from_pose_to_matrix(self, pose):
        t = pose.xyz()
        q = pose.quatertion()
        rotation = self._from_quaternion_to_matrix(*q)
        M = np.eye(4)
        M[0:3,0:3] = rotation
        M[0,3] = t[0]
        M[1,3] = t[1]
        M[2,3] = t[2]
        return M

    def _from_matrix_to_transform(self, M):
        R = np.linalg.inv( M[0:3,0:3])
        H = np.eye(4)
        H[0:3,0:3] = R
        H[0:3,3] = -R.dot(M[0:3,3])
        return H
    
    def _transform_position_of_fromGto_(self, pose, idx):
        M_rInG = self._from_pose_to_matrix( pose )
        p_rInG = M_rInG[0:3,3]
        M_IdxInG = self.init_matrices[idx]
        p_IdxInG = M_IdxInG[0:3,3]
        H_GToIdx = self.init_transforms[idx] 
        p_rInIdx = H_GToIdx[0:3,0:3].dot(p_rInG-p_IdxInG).flatten()
        return p_rInIdx 

    def _transform_orientation_of_fromGto_(self, pose, idx):
        M_rInG = self._from_pose_to_matrix( pose )
        M_IdxInG = self.init_matrices[idx]
        rotation_rInIdx = np.linalg.inv( M_IdxInG[0:3,0:3]).dot( M_rInG[0:3,0:3])
        q_rInIdx = np.array( self._from_matrix_to_quaternion(rotation_rInIdx) ).flatten()
        return q_rInIdx 

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
        for idx, (cppr,r) in enumerate(zip(self.cpp_robots,self.robot_bodies) ):
            pose = r.pose()    
            pr_poses[cppr] = np.concatenate( [ self._transform_position_of_fromGto_( pose, idx), self._transform_orientation_of_fromGto_(pose, idx) ]).flatten() 
        
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
            # joints: nbr_joints(==2 * nbr_legs == 8) * 2 (position and velocity) ==  16
            obs[idxr].append( self.state["contacts"][cppr])
            # contacts: nbr_parts*1 == 20 
            for cppo, o in zip(self.cpp_robots, self.robot_bodies):
                if cppr != cppo :
                    pose = o.pose()
                    pose_oInr = np.concatenate( [ self._transform_position_of_fromGto_( pose, idxr), self._transform_orientation_of_fromGto_(pose, idxr) ]).flatten() 
                    obs[idxr].append( pose_oInr ) 
                # opponent pose: x,y,z, qx,qy,qz,qw : 7
            obs[idxr] = np.reshape( np.concatenate(obs[idxr]).flatten(), newshape=(1,-1) )
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

        pr_a = np.split(a, 2, axis=0)
        pr_ctrl_cost = []
        pr_attack_reward = []
        for i in range(len(self.cpp_robots)):
            pr_ctrl_cost.append( self._compute_control_cost(pr_a[i]) )
            pr_attack_reward.append( self._compute_attack_reward(i) )
        
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
        for alive, el_cost, limit_cost, ctrl_cost, attack_reward in zip(alives,pr_electricity_cost,pr_joints_at_limit_cost, pr_ctrl_cost, pr_attack_reward ):
            #self.dense_rewards.append( sum([alive, el_cost, limit_cost, ctrl_cost, attack_reward]) )
            self.dense_rewards.append( sum([attack_reward]) )


        self.frame  += 1
        if (done and not self.done) or self.frame==self.spec.timestep_limit:
            self.done = True 
            self.episode_over(self.frame)
        else :
            self.done = done

        if self.use_reward_shaping:
            self.rewards = self.dense_rewards
        else :
            self.rewards = [0.0 for _ in self.cpp_robots]
        if self.done:
            if draw : 
                self.rewards = [-1000 for _ in self.cpp_robots]
            else :
                for idx, alive in enumerate(alives):
                    pr_reward = 2000.0 if alive else -2000.0
                    self.rewards[idx] = pr_reward
                
        for i in range(len(self.cpp_robots)):
            infos[i]['dense_reward'] = self.dense_rewards[i]
        
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

