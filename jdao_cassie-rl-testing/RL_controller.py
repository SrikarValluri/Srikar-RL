from cassie.cassiemujoco.cassieUDP import *
from cassie.cassiemujoco.cassiemujoco import CassieSim
from cassie.cassiemujoco.cassiemujoco_ctypes import *
# from cassie.speed_env import CassieEnv_speed
# from cassie.speed_double_freq_env import CassieEnv_speed_dfreq
# from cassie.speed_no_delta_env import CassieEnv_speed_no_delta
# from cassie.speed_no_delta_neutral_foot_env import CassieEnv_speed_no_delta_neutral_foot

import time
import numpy as np
import matplotlib.pyplot as plt
import pylab
import torch
import torch.nn
import pickle
from rl.policies import GaussianMLP
import platform
from quaternion_function import *

#import signal 
import atexit
import sys
import datetime

time_log   = [] # time stamp
input_log  = [] # network inputs
output_log = [] # network outputs 
state_log  = [] # cassie state
target_log = [] #PD target log

#PREFIX = "./"
PREFIX = "/home/drl/jdao/jdao_cassie-rl-testing/"


# if len(sys.argv) > 1:
#     filename = PREFIX + "logs/" + sys.argv[1]
# else:
#     filename = PREFIX + "logs/" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M')


max_speed = 1.5
min_speed = 0.0

# def log(sto="final"):
#     data = {"time": time_log, "output": output_log, "input": input_log, "state": state_log, "target": target_log}

#     filep = open(filename + "_log" + sto + ".pkl", "wb")

#     pickle.dump(data, filep)

#     filep.close()

# atexit.register(log)

# Prevent latency issues by disabling multithreading in pytorch
torch.set_num_threads(1)

# Prepare model
# env = CassieEnv_speed_no_delta_neutral_foot("walking", clock_based=True, state_est=True)
# env.reset_for_test()
phase = 0
counter = 0 
phase_add = 1
speed = 0



policy = torch.load('/home/beast/srikar/jdao_cassie-rl-testing/trained_models/nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2.pt')
policy.eval()

max_speed = 3.0
min_speed = -1.0
max_y_speed = 0.0
min_y_speed = 0.0
symmetry = True

# Initialize control structure with gains
P = np.array([100, 100, 88, 96, 50, 100, 100, 88, 96, 50])
D = np.array([10.0, 10.0, 8.0, 9.6, 5.0, 10.0, 10.0, 8.0, 9.6, 5.0])
u = pd_in_t()
for i in range(5):
    u.leftLeg.motorPd.pGain[i] = P[i]
    u.leftLeg.motorPd.dGain[i] = D[i]
    u.rightLeg.motorPd.pGain[i] = P[i+5]
    u.rightLeg.motorPd.dGain[i] = D[i+5]

pos_index = np.array([2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])
pos_mirror_index = np.array([2,3,4,5,6,21,22,23,28,29,30,34,7,8,9,14,15,16,20])
vel_mirror_index = np.array([0,1,2,3,4,5,19,20,21,25,26,27,31,6,7,8,12,13,14,18])
offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

# Determine whether running in simulation or on the robot
if platform.node() == 'cassie':
    cassie = CassieUdp(remote_addr='10.10.10.3', remote_port='25010',
                       local_addr='10.10.10.100', local_port='25011')
else:
    cassie = CassieUdp() # local testing
    

# Connect to the simulator or robot
print('Connecting...')
y = None
while y is None:
    cassie.send_pd(pd_in_t())
    time.sleep(0.001)
    y = cassie.recv_newest_pd()
received_data = True
print('Connected!\n')

# Record time
t = time.monotonic()
t0 = t

# Whether or not STO has been TOGGLED (i.e. it does not count the initial STO condition)
sto = False
sto_count = 0

orient_add = 0

i_vector = []
pel_x = []
pel_y = []
pel_z = []
action_pos_0 = []
action_pos_1 = []
action_pos_2 = []
action_pos_3 = []
action_pos_4 = []
action_pos_5 = []
action_pos_6 = []
action_pos_7 = []
action_pos_8 = []
action_pos_9 = []
motor_pos_0 = []
motor_pos_1 = []
motor_pos_2 = []
motor_pos_3 = []
motor_pos_4 = []
motor_pos_5 = []
motor_pos_6 = []
motor_pos_7 = []
motor_pos_8 = []
motor_pos_9 = []
motor_torque_0 = []
motor_torque_1 = []
motor_torque_2 = []
motor_torque_3 = []
motor_torque_4 = []
motor_torque_5 = []
motor_torque_6 = []
motor_torque_7 = []
motor_torque_8 = []
motor_torque_9 = []
counter_not_used_elsewhere = 0

sim = CassieSim("./cassie/cassiemujoco/cassie.xml")

while True:
#     if counter_not_used_elsewhere == 0:
#         sim.set_qpos([0.0, 0.0, 1.01, 0.8, 0.0, 0.0, 0.0, 0.0045, 
#         0.0, 0.4973, 0.9784830934748516, -0.016399716640763992, 0.017869691242100763, 
#         -0.2048964597373501, -1.1997, 0.0, 1.4267, 0.0, -1.5244, 1.5244, -1.5968, -0.0045, 
#         0.0, 0.4973, 0.978614127766972, 0.0038600557257107214, -0.01524022001550036, 
#         -0.20510296096975877, -1.1997, 0.0, 1.4267, 0.0, -1.5244, 1.5244, -1.5968]
# )
    print(sim.qpos())
    # Wait until next cycle time
    while time.monotonic() - t < 60/2000:
        time.sleep(0.001)
    t = time.monotonic()
    tt = time.monotonic() - t0

    # Get newest state
    state = cassie.recv_newest_pd()

    if state is None:
        print('Missed a cycle')
        continue	

    if platform.node() == 'cassie':
        # Radio control
        orient_add -= state.radio.channel[3] / 60.0

        # Reset orientation on STO
        if state.radio.channel[8] < 1:
            orient_add = quaternion2euler(state.pelvis.orientation[:])[2]

            # Save log files after STO toggle (skipping first STO)
            if sto is False:
                log(sto_count)
                sto_count += 1
                sto = True
        else:
            sto = False
        
        curr_max = max_speed / 2# + (max_speed / 2)*state.radio.channel[4]
        #print("curr_max:", curr_max)
        speed_add = (max_speed / 2) * state.radio.channel[4]
        speed = max(min_speed, state.radio.channel[0] * curr_max + speed_add)
        speed = min(max_speed, state.radio.channel[0] * curr_max + speed_add)
        
        print("speed: ", speed)
        phase_add = 1+state.radio.channel[5]
        # env.y_speed = max(min_y_speed, -state.radio.channel[1] * max_y_speed)
        # env.y_speed = min(max_y_speed, -state.radio.channel[1] * max_y_speed)
    else:
        # Automatically change orientation and speed
        tt = time.monotonic() - t0
        orient_add += 0#math.sin(t / 8) / 400
        #env.speed = 0.2
        speed += 0.001#((math.sin(tt / 2)) * max_speed)
        print("speed: ", speed)
        #if env.phase % 14 == 0:
        #	env.speed = (random.randint(-1, 1)) / 2.0
        # print(env.speed)
        speed = max(min_speed, speed)
        speed = min(max_speed, speed)
        # env.y_speed = (math.sin(tt / 2)) * max_y_speed
        # env.y_speed = max(min_y_speed, env.y_speed)
        # env.y_speed = min(max_y_speed, env.y_speed)

    # if env.phase < 14 or symmetry is False:
    	# quaternion = euler2quat(z=env.orientation, y=0, x=0)
    	# iquaternion = inverse_quaternion(quaternion)
    	# new_orientation = quaternion_product(iquaternion, state.pelvis.orientation[:])
    # 	if new_orientation[0] < 0:
    # 		new_orientation = -new_orientation
    # 	new_translationalVelocity = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)

    # 	print('quaternion: {}, new_orientation: {}'.format(quaternion, new_orientation))

    # 	# Construct input vector
    # 	if symmetry:
    # 		cassie_state = np.copy(np.concatenate([[state.pelvis.position[2] - state.terrain.height], new_orientation[:], state.motor.position[:], new_translationalVelocity[:], state.pelvis.rotationalVelocity[:], state.motor.velocity[:], state.pelvis.translationalAcceleration[:], state.joint.position[:], state.joint.velocity[:]]))
    # 	else:
    # 		cassie_state = np.copy(np.concatenate([[state.pelvis.position[2] - state.terrain.height], new_orientation[:], state.motor.position[:], new_translationalVelocity[:], state.pelvis.rotationalVelocity[:], state.motor.velocity[:], state.pelvis.translationalAcceleration[:], state.leftFoot.toeForce[:], state.leftFoot.heelForce[:], state.rightFoot.toeForce[:], state.rightFoot.heelForce[:]]))
    # 	ref_pos, ref_vel = env.get_kin_next_state()
    # 	RL_state = np.concatenate([cassie_state, ref_pos[pos_index], ref_vel[vel_index]])
    # else:
    # 	quaternion = euler2quat(z=env.orientation, y=0, x=0)
    # 	cassie_state = get_mirror_state(state, quaternion)
    # 	ref_pos, ref_vel = env.get_kin_next_state()
    # 	ref_vel[1] = -ref_vel[1]
    # 	euler = quaternion2euler(ref_pos[3:7])
    # 	euler[0] = -euler[0]
    # 	euler[2] = -euler[2]
    # 	ref_pos[3:7] = euler2quat(z=euler[2],y=euler[1],x=euler[0])
    # 	RL_state = np.concatenate([cassie_state, ref_pos[pos_mirror_index], ref_vel[vel_mirror_index]])


    clock = [np.sin(2 * np.pi *  phase / 27), np.cos(2 * np.pi *  phase / 27)]
    # euler_orient = quaternion2euler(state.pelvis.orientation[:]) 
    # print("euler orient: ", euler_orient + np.array([orient_add, 0, 0]))
    # new_orient = euler2quat(euler_orient + np.array([orient_add, 0, 0]))
    quaternion = euler2quat(z=orient_add, y=0, x=0)
    iquaternion = inverse_quaternion(quaternion)
    new_orient = quaternion_product(iquaternion, state.pelvis.orientation[:])
    if new_orient[0] < 0:
        new_orient = -new_orient
    new_translationalVelocity = rotate_by_quaternion(state.pelvis.translationalVelocity[:], iquaternion)
    print('new_orientation: {}'.format(new_orient))
        
    ext_state = np.concatenate([clock, [speed]])
    robot_state = np.concatenate([
            [state.pelvis.position[2] - state.terrain.height], # pelvis height
            # new_orient,
            state.pelvis.orientation[:],                                 # pelvis orientation
            state.motor.position[:],                                     # actuated joint positions

            state.pelvis.translationalVelocity[:],                       # pelvis translational velocity
            # new_translationalVelocity[:],
            state.pelvis.rotationalVelocity[:],                          # pelvis rotational velocity 
            state.motor.velocity[:],                                     # actuated joint velocities

            state.pelvis.translationalAcceleration[:],                   # pelvis translational acceleration
            
            state.joint.position[:],                                     # unactuated joint positions
            state.joint.velocity[:]                                      # unactuated joint velocities
    ])
    RL_state = np.concatenate([robot_state, ext_state])

    #pretending the height is always 1.0
    # RL_state[0] = 1.0
    
    # Construct input vector
    torch_state = torch.Tensor(RL_state)
    # torch_state = shared_obs_stats.normalize(torch_state)

    # Get action
    _, action = policy.act(torch_state, True)

    env_action = action.data.numpy()
    target = np.ndarray.flatten(env_action + offset)
    print(target)
    #print(state.pelvis.position[2] - state.terrain.height)

    # Send action
    for i in range(5):
        u.leftLeg.motorPd.pTarget[i] = target[i]
        u.rightLeg.motorPd.pTarget[i] = target[i+5]
    #time.sleep(0.005)
    cassie.send_pd(u)

    # Measure delay
    print('delay: {:6.1f} ms'.format((time.monotonic() - t) * 1000))

    # Logging
    time_log.append(time.time())
    state_log.append(state)
    input_log.append(RL_state)
    output_log.append(env_action)
    target_log.append(target)

    # Track phase
    phase += phase_add
    if phase >= 28:
        phase = 0
        counter += 1

    i_vector.append(counter_not_used_elsewhere)
    pel_x.append(state.pelvis.position[0])
    pel_y.append(state.pelvis.position[1])
    pel_z.append(state.pelvis.position[2])
    motor_pos_0.append(state.motor.position[0])
    action_pos_0.append(action[0])
    motor_pos_1.append(state.motor.position[1])
    action_pos_1.append(action[1])
    motor_pos_2.append(state.motor.position[2])
    action_pos_2.append(action[2])
    motor_pos_3.append(state.motor.position[3])
    action_pos_3.append(action[3])
    motor_pos_4.append(state.motor.position[4])
    action_pos_4.append(action[4])
    motor_pos_5.append(state.motor.position[5])
    action_pos_5.append(action[5])
    motor_pos_6.append(state.motor.position[6])
    action_pos_6.append(action[6])
    motor_pos_7.append(state.motor.position[7])
    action_pos_7.append(action[7])
    motor_pos_8.append(state.motor.position[8])
    action_pos_8.append(action[8])
    motor_pos_9.append(state.motor.position[9])
    action_pos_9.append(action[9])

    motor_torque_0.append(state.motor.torque[0])
    motor_torque_1.append(state.motor.torque[1])
    motor_torque_2.append(state.motor.torque[2])
    motor_torque_3.append(state.motor.torque[3])
    motor_torque_4.append(state.motor.torque[4])
    motor_torque_5.append(state.motor.torque[5])
    motor_torque_6.append(state.motor.torque[6])
    motor_torque_7.append(state.motor.torque[7])
    motor_torque_8.append(state.motor.torque[8])
    motor_torque_9.append(state.motor.torque[9])
    counter_not_used_elsewhere += 1

# plt.title('Torques vs. Time (t)')
# plt.plot(i_vector, motor_torque_0, label = '0')
# plt.plot(i_vector, motor_torque_1, label = '1')
# plt.plot(i_vector, motor_torque_2, label = '2')
# plt.plot(i_vector, motor_torque_3, label = '3')
# plt.plot(i_vector, motor_torque_4, label = '4')
# plt.plot(i_vector, motor_torque_5, label = '5')
# plt.plot(i_vector, motor_torque_6, label = '6')
# plt.plot(i_vector, motor_torque_7, label = '7')
# plt.plot(i_vector, motor_torque_8, label = '8')
# plt.plot(i_vector, motor_torque_9, label = '9')

# plt.xlabel('Timesteps')
# plt.ylabel('Torque (N*m)')
# plt.legend(loc='upper right')
# plt.axis([-20,400,-30,30])

# plt.show()    


plt.figure(1)
plt.title('X, Y, and Z Pelvis Positions vs. Time')
x = plt.plot(i_vector, pel_x, 'r-', label = 'x')
y = plt.plot(i_vector, pel_y, 'g-', label = 'y')
z = plt.plot(i_vector, pel_z, 'b-', label = 'z')
plt.xlabel('Timesteps (t)')
plt.ylabel('Position (x, y, z)')
plt.legend(loc='upper left')



plt.figure(2)
plt.title('Motor Position 0, 5 vs. Time')
plt.plot(i_vector, motor_pos_0, 'r-', label = 'Motor 0')
plt.plot(i_vector, action_pos_0, 'g-', label = 'Action 0')
plt.plot(i_vector, motor_pos_5, 'b-', label = 'Motor 5')
plt.plot(i_vector, action_pos_5, 'm-', label = 'Action 5')
plt.xlabel('Timesteps (t)')
plt.ylabel('Motor Position 0, 5')
plt.legend(loc='upper right')

plt.figure(3)
plt.title('Motor Position 1, 6 vs. Time')
plt.plot(i_vector, motor_pos_1, 'r-', label = 'Motor 1')
plt.plot(i_vector, action_pos_1, 'g-', label = 'Action 1')
plt.plot(i_vector, motor_pos_6, 'b-', label = 'Motor 6')
plt.plot(i_vector, action_pos_6, 'm-', label = 'Action 6')
plt.xlabel('Timesteps (t)')
plt.ylabel('Motor Position 1, 6')
plt.legend(loc='upper right')

plt.figure(4)
plt.title('Motor Position 2, 7 vs. Time')
plt.plot(i_vector, motor_pos_2, 'r-', label = 'Motor 2')
plt.plot(i_vector, action_pos_2, 'g-', label = 'Action 2')
plt.plot(i_vector, motor_pos_7, 'b-', label = 'Motor 7')
plt.plot(i_vector, action_pos_7, 'm-', label = 'Action 7')
plt.xlabel('Timesteps (t)')
plt.ylabel('Motor Position 2, 7')
plt.legend(loc='upper right')

plt.figure(5)
plt.title('Motor Position 3, 8 vs. Time')
plt.plot(i_vector, motor_pos_3, 'r-', label = 'Motor 3')
plt.plot(i_vector, action_pos_3, 'g-', label = 'Action 3')
plt.plot(i_vector, motor_pos_8, 'b-', label = 'Motor 8')
plt.plot(i_vector, action_pos_8, 'm-', label = 'Action 8')
plt.xlabel('Timesteps (t)')
plt.ylabel('Motor Position 3, 8')
plt.legend(loc='upper right')

plt.figure(6)
plt.title('Motor Position 4, 9 vs. Time')
plt.plot(i_vector, motor_pos_4, 'r-', label = 'Motor 4')
plt.plot(i_vector, action_pos_4, 'g-', label = 'Action 4')
plt.plot(i_vector, motor_pos_9, 'b-', label = 'Motor 9')
plt.plot(i_vector, action_pos_9, 'm-', label = 'Action 9')
plt.xlabel('Timesteps (t)')
plt.ylabel('Motor Position 4, 9')
plt.legend(loc='upper right')

plt.title('Torques vs. Time (t)')
plt.plot(i_vector, motor_torque_0, label = '0')
plt.plot(i_vector, motor_torque_1, label = '1')
plt.plot(i_vector, motor_torque_2, label = '2')
plt.plot(i_vector, motor_torque_3, label = '3')
plt.plot(i_vector, motor_torque_4, label = '4')
plt.plot(i_vector, motor_torque_5, label = '5')
plt.plot(i_vector, motor_torque_6, label = '6')
plt.plot(i_vector, motor_torque_7, label = '7')
plt.plot(i_vector, motor_torque_8, label = '8')
plt.plot(i_vector, motor_torque_9, label = '9')

plt.xlabel('Timesteps')
plt.ylabel('Torque (N*m)')
plt.legend(loc='upper right')
plt.axis([-20,400,-30,30])

plt.show()