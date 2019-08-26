from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

from .trajectory import CassieTrajectory

from math import floor

import numpy as np 
import os
import random

import gym
from gym import spaces
import pickle

class CassieMimicEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, traj="walking", simrate=60, clock_based=False):

        self.sim = CassieSim()
        self.vis = None

        # NOTE: Xie et al uses full reference trajectory info
        # (i.e. clock_based=False)
        self.clock_based = clock_based


        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(48,))
        # self.action_space      = spaces.Box(low=-np.inf, high=np.inf, shape=(10,) )

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(46,))
        self.action_space      = spaces.Box(low=-np.inf, high=np.inf, shape=(10,))

        self.qpos0 = np.copy(self.sim.qpos())
        self.qvel0 = np.copy(self.sim.qvel()) 

        self.random_trajectories = pickle.load(open('/home/beast/srikar/gym-cassie-master/gym_cassie/envs/spline_stepping_traj.pkl', 'rb'))

        dirname = os.path.dirname(__file__)
        if traj == "walking":
            traj_path = os.path.join(dirname, "trajectory", "stepdata.bin")

        elif traj == "stand-in-place":
            raise NotImplementedError

        self.trajectory = CassieTrajectory(traj_path)

        self.P = np.array([100,  100,  88,  96,  50]) 
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        self.u = pd_in_t()

        self.cassie_state = state_out_t()

        self.simrate = simrate # simulate X mujoco steps with same pd target
                               # 60 brings simulation from 2000Hz to roughly 30Hz

        self.time    = 0 # number of time steps in current episode
        self.phase   = 0 # portion of the phase the robot is in
        self.counter = 0 # number of phase cycles completed in episode

        # NOTE: a reference trajectory represents ONE phase cycle

        # should be floor(len(traj) / simrate) - 1
        # should be VERY cautious here because wrapping around trajectory
        # badly can cause assymetrical/bad gaits
        self.phaselen = floor(len(self.trajectory) / self.simrate) - 1

        # see include/cassiemujoco.h for meaning of these indices
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]


        self.stand_pos = np.array([ 0.,  0.    ,     1.01   ,  1.        ,  0.        ,
        0.        ,  0.        ,  0.0045    ,  0.        ,  0.4973    ,
        0.97848309, -0.01639972,  0.01786969, -0.20489646, -1.1997    ,
        0.        ,  1.4267    ,  0.        , -1.5244    ,  1.5244    ,
       -1.5968    , -0.0045    ,  0.        ,  0.4973    ,  0.97861413,
        0.00386006, -0.01524022, -0.20510296, -1.1997    ,  0.        ,
        1.4267    ,  0.        , -1.5244    ,  1.5244    , -1.5968    ])

        self.stand_vel = np.zeros(35)

        self.stand_state = np.array([
            1.01, 1, 0, 0, 0, 0.0045, 0, 0.4973, -1.1997, -1.5968, 
            -0.0045, 0, 0.4973, -1.1997, -1.5968, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
            0, 0, 0, 0, 0, 1.4267, -1.5968, 0, 1.4267, -1.5968, 
            0, 0, 0, 0, 0, 0
        ])

        self.action_queue = []
        self.action_time = 0

        self.height_diff = 0
        self.pel_vel = 0
        self.motor_torque = 0
        self.vel_diff = 0
        self.theta_diff = 0

    @property
    def dt(self):
        return 1 / 2000 * self.simrate

    def close(self):
        if self.vis is not None:
            del self.vis # overloaded to call cassie_vis_free
            self.vis = None
    
    def step_simulation(self, action, policy_name):
        ref_pos, ref_vel = self.get_ref_state(self.phase + 1)

        if(policy_name == 'step'):
            target = action + ref_pos[self.pos_idx]
        else:
            offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
            target = action + offset
        # target[4] += -1.5968
        # target[9] += -1.5968
        
        # target = action + ref_pos[self.pos_idx]
        
        self.u = pd_in_t()
        self.sim.apply_force([np.random.uniform(-30, 30), np.random.uniform(-30, 30), 0, 0, 0])
        for i in range(5):
            # TODO: move setting gains out of the loop?
            # maybe write a wrapper for pd_in_t ?
            self.u.leftLeg.motorPd.pGain[i]  = self.P[i]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i]

            self.u.leftLeg.motorPd.dGain[i]  = self.D[i]
            self.u.rightLeg.motorPd.dGain[i] = self.D[i]

            self.u.leftLeg.motorPd.torque[i]  = 0 # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0 

            self.u.leftLeg.motorPd.pTarget[i]  = target[i]
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        self.cassie_state = self.sim.step_pd(self.u)

    def step(self, action):
        policy_name = 'stand'
        if self.action_time == 0:
            for i in range(3):
                self.action_queue.append(action)
        else:
            self.action_queue.insert(0, action)
        delayed_action = self.action_queue.pop()
        for _ in range(self.simrate):
            self.step_simulation(delayed_action, policy_name)

        height = self.sim.qpos()[2]

        self.time  += 1
        self.phase += 1
        self.action_time += 1

        if self.phase > self.phaselen:
            self.phase = 0
            self.counter += 1

        # Early termination
        done = not(height > 0.4 and height < 3.0)


        reward = self.compute_reward(policy_name)

        # TODO: make 0.3 a variable/more transparent
        if reward < 0.3:
            done = True

        return self.get_full_state(policy_name), reward, done, {}

    def reset(self):
        policy_name = 'stand'
        self.phase = random.randint(0, self.phaselen)
        self.time = 0
        self.counter = 0
        if(policy_name == 'step'):
            qpos, qvel = self.get_ref_state(self.phase)
        else:
            # height = np.random.uniform(0.80, 1.2)
            # self.qpos0[2] = height
            x = np.random.choice(self.random_trajectories['qpos'].shape[0], size = 1, replace = False)
            y = np.random.choice(self.random_trajectories['qvel'].shape[0], size = 1, replace = False)
            self.qpos0, self.qvel0 = self.random_trajectories['qpos'][x], self.random_trajectories['qvel'][y]
            qpos, qvel = self.qpos0, self.qvel0
            self.prev_state = self.get_full_state('stand')
        self.sim.set_qpos(np.squeeze(qpos))
        self.sim.set_qvel(np.squeeze(qvel))

        return self.get_full_state(policy_name)

    # used for plotting against the reference trajectory
    def reset_for_test(self):
        self.phase = 0
        self.time = 0
        self.counter = 0


        qpos, qvel = self.get_full_state_seperated()

        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)

        return self.get_full_state()
    
    def set_joint_pos(self, jpos, fbpos=None, iters=5000): #self, jpos = 1 x 20 random 0 - pi, fbpos = 1 x 7 with first 3 random and last 4 between 0 and 1, magnitude of whole vector has to be 1
        """
        Kind of hackish. 
        This takes a floating base position and some joint positions
        and abuses the MuJoCo solver to get the constrained forward
        kinematics. 

        There might be a better way to do this, e.g. using mj_kinematics
        """

        # actuated joint indices
        joint_idx = [7, 8, 9, 14, 20,
                     21, 22, 23, 28, 34]

        # floating base indices
        fb_idx = [0, 1, 2, 3, 4, 5, 6]

        for _ in range(iters):
            qpos = np.copy(self.sim.qpos())
            qvel = np.copy(self.sim.qvel())

            qpos[joint_idx] = jpos

            if fbpos is not None:
                qpos[fb_idx] = fbpos

            self.sim.set_qpos(qpos)
            self.sim.set_qvel(0 * qvel)

            self.sim.step_pd(pd_in_t())


    # NOTE: this reward is slightly different from the one in Xie et al
    # see notes for details
    def compute_reward(self, policy_name):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())

        # stand_pos = np.array([0.0, 0.0, 1.01, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.978483, -0.016400, 0.017870, -0.204896, 0.0, 0.0, 1.426700, 0.0, 
        # -1.524400, 1.524400, 0.0, 0.0, 0.0, 0.0, 0.978614, 0.003860, -0.015240, -0.205103, 0.0, 0.0, 1.426700, 0.0, -1.524400, 1.524400, 0.0])
        if policy_name == 'stand':
            ref_pos, ref_vel = self.get_ref_state(self.phase)

            weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]

            joint_error       = 0
            com_error         = 0
            orientation_error = 0
            spring_error      = 0

            # each joint pos
            for i, j in enumerate(self.pos_idx):
                target = ref_pos[j]
                actual = qpos[j]

                joint_error += 30 * weight[i] * (target - actual) ** 2

            # center of mass: x, y, z
            for j in [0, 1, 2]:
                target = ref_pos[j]
                actual = qpos[j]

                # NOTE: in Xie et al y target is 0

                com_error += (target - actual) ** 2
            
            # COM orientation: qx, qy, qz
            for j in [4, 5, 6]:
                target = ref_pos[j] # NOTE: in Xie et al orientation target is 0
                actual = qpos[j]

                orientation_error += (target - actual) ** 2

            # left and right shin springs
            for i in [15, 29]:
                target = ref_pos[i] # NOTE: in Xie et al spring target is 0
                actual = qpos[i]

                spring_error += 1000 * (target - actual) ** 2      
            
            reward = 0.5 * np.exp(-joint_error) +       \
                    0.3 * np.exp(-com_error) +         \
                    0.1 * np.exp(-orientation_error) + \
                    0.1 * np.exp(-spring_error)

            return reward
        else:
            stand_pos = np.array([0.0, 0.0, 1.01, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.978483, -0.016400, 0.017870, -0.204896, 0.0, 0.0, 1.426700, 0.0, 
            -1.524400, 1.524400, 0.0, 0.0, 0.0, 0.0, 0.978614, 0.003860, -0.015240, -0.205103, 0.0, 0.0, 1.426700, 0.0, -1.524400, 1.524400, 0.0])
            
            self.height_diff = np.linalg.norm(qpos[7:] - stand_pos[7:])
            self.pel_vel = np.linalg.norm(qvel[0:3])
            # self.motor_torque = np.linalg.norm(self.cassie_state.motor.torque[:])
            # self.vel_diff = np.linalg.norm(np.concatenate([qvel[6:9], qvel[12], qvel[18:22], qvel[25], qvel[31]]))
            self.theta_diff = (np.abs(np.arccos(2 * self.stand_pos[3] ** 2 * qpos[3] ** 2 - 1))) ** 2
            # self.accel_diff = np.linalg.norm(self.cassie_state.pelvis.translationalAcceleration[:])
            # reward = 0.1*np.exp(-self.height_diff) + 0.15*np.exp(-self.pel_vel) + 0.05*np.exp(-0.01 * self.motor_torque) + 0.05*np.exp(-self.vel_diff) + 0.1*np.exp(-self.theta_diff) + 0.05*np.exp(-self.accel_diff)

            # loss = np.linalg.norm(self.stand_state - self.get_full_state())
            # reward = 0.5 * np.exp(-loss)

            loss = 0.3 * self.height_diff + 0.4 * self.pel_vel + 0.3 * self.theta_diff
            reward = 0.5 * np.exp(-loss)
            return reward

    # get the corresponding state from the reference trajectory for the current phase
    def get_ref_state(self, phase=None):
        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        pos = np.copy(self.trajectory.qpos[phase * self.simrate])

        # this is just setting the x to where it "should" be given the number
        # of cycles
        pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter
        
        # ^ should only matter for COM error calculation,
        # gets dropped out of state variable for input reasons

        # setting lateral distance target to 0
        pos[1] = 0

        vel = np.copy(self.trajectory.qvel[phase * self.simrate])

        return pos, vel

    def get_full_state_seperated(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel()) 

        return qpos, qvel
        
    def get_full_state(self, policy_name):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel()) 

        ref_pos, ref_vel = self.get_ref_state(self.phase + 1)

        # this is everything except pelvis x and qw, achilles rod quaternions, 
        # and heel spring/foot crank/plantar rod angles
        # NOTE: x is forward dist, y is lateral dist, z is height

        # makes sense to always exclude x because it is in global coordinates and
        # irrelevant to phase-based control. Z is inherently invariant to (flat)
        # trajectories despite being global coord. Y is only invariant to straight
        # line trajectories.

        # [ 0] Pelvis y
        # [ 1] Pelvis z
        # [ 2] Pelvis orientation qw
        # [ 3] Pelvis orientation qx
        # [ 4] Pelvis orientation qy
        # [ 5] Pelvis orientation qz
        # [ 6] Left hip roll         (Motor [0])
        # [ 7] Left hip yaw          (Motor [1])
        # [ 8] Left hip pitch        (Motor [2])
        # [ 9] Left knee             (Motor [3])
        # [10] Left shin                        (Joint [0])
        # [11] Left tarsus                      (Joint [1])
        # [12] Left foot              (Motor [4], Joint [2])
        # [13] Right hip roll        (Motor [5])
        # [14] Right hip yaw         (Motor [6])
        # [15] Right hip pitch       (Motor [7])
        # [16] Right knee            (Motor [8])
        # [17] Right shin                       (Joint [3])
        # [18] Right tarsus                     (JCassieIKTrajectoryoint [4])
        # [19] Right foot            (Motor [9], JCassieIKTrajectoryoint [5])
        pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])

        # [ 0] Pelvis x
        # [ 1] Pelvis y
        # [ 2] Pelvis z
        # [ 3] Pelvis orientation wx
        # [ 4] Pelvis orientation wy
        # [ 5] Pelvis orientation wz
        # [ 6] Left hip roll         (Motor [0])
        # [ 7] Left hip yaw          (Motor [1])
        # [ 8] Left hip pitch        (Motor [2])
        # [ 9] Left knee             (Motor [3]) 
        # [10] Left shin                        
        # [11] Left tarsus                      
        # [12] Left foot             (Motor [4],
        # [13] Right hip roll        (Motor [5])
        # [14] Right hip yaw         (Motor [6])
        # [15] Right hip pitch       (Motor [7])
        # [16] Right knee            (Motor [8])
        # [17] Right shin                       (Joint [3])
        # [18] Right tarsus                     (Joint [4])
        # [19] Right foot            (Motor [9], Joint [5])
        vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        clock = [np.sin(2 * np.pi *  self.phase / self.phaselen),
                 np.cos(2 * np.pi *  self.phase / self.phaselen)]
            
        ext_state = clock


        # ext_state = np.concatenate([ref_pos[pos_index], ref_vel[vel_index]])


        robot_state = np.concatenate([
            [self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height], # pelvis height 1
            self.cassie_state.pelvis.orientation[:],                                 # pelvis orientation 4 
            self.cassie_state.motor.position[:],                                     # actuated joint positions 10

            self.cassie_state.pelvis.translationalVelocity[:],                       # pelvis translational velocity 3
            self.cassie_state.pelvis.rotationalVelocity[:],                          # pelvis rotational velocity 3
            self.cassie_state.motor.velocity[:],                                     # actuated joint velocities 10

            self.cassie_state.pelvis.translationalAcceleration[:],                   # pelvis translational acceleration 3
            
            self.cassie_state.joint.position[:],                                     # unactuated joint positions 6
            self.cassie_state.joint.velocity[:]                                      # unactuated joint velocities 6
        ])

        # robot_state = np.concatenate([self.sim.qpos(), self.sim.qvel()])

        if(policy_name == 'step'):
            return np.concatenate([robot_state, ext_state])
        else:
            self.prev_state = robot_state
            return np.concatenate([robot_state])
    
    def render(self):
        if self.vis is None:
            self.vis = CassieVis()

        self.vis.draw(self.sim)


















