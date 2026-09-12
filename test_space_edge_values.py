import  cart_model
from tqdm.auto import tqdm
import policies
import random
import tqdm as tqdm
random.seed(0)
context = cart_model.Cart_model(human=False)
episodes = 1000


#keep bot positive and negative values for the velocity, angle, position and angular velocity
max_velocity = [0,0]
min_velocity = [999999, 999999]
min_angle = [999999, 999999]
max_angle = [0, 0]
max_angular_velocity = [0, 0]
min_angular_velocity = [9999999, 9999999]
max_position =[0,0]



for i in range(10000):
    episode = cart_model.Episode()
    while episode.terminated == False:    
        state,action,reward=episode.one_step(context, None)
        #print("position: ", state.position, "velocity: ", state.velocity, "angle: ", state.pole_angle, "angular velocity: ", state.pole_angle_velocity)
        if state.velocity > 0 :
            if abs(state.velocity) > abs(max_velocity[0]):
                max_velocity[0] = state.velocity
            if abs(state.velocity) < abs(min_velocity[0]):
                min_velocity[0] = state.velocity
        else:
            if abs(state.velocity) >abs(max_velocity[1]):
                max_velocity[1] = state.velocity
            if abs(state.velocity) < abs(min_velocity[1]):
                min_velocity[1] = state.velocity
        if state.position> max_position[0]:
            max_position[0]=state.position
        elif state.position< max_position[1]:
            max_position[1]=state.position
        if state.pole_angle > 0:
            if abs(state.pole_angle) > abs(max_angle[0]):
                max_angle[0] = state.pole_angle
            if abs(state.pole_angle) < abs(min_angle[0]):
                min_angle[0] = state.pole_angle
        else:
            if abs(state.pole_angle) > abs(max_angle[1]):
                max_angle[1] = state.pole_angle
            if abs(state.pole_angle) < abs(min_angle[1]):
                min_angle[1] = state.pole_angle
        if state.pole_angle_velocity > 0:
            if abs(state.pole_angle_velocity) > abs(max_angular_velocity[0]):
                max_angular_velocity[0] = state.pole_angle_velocity
            if abs(state.pole_angle_velocity) < abs(min_angular_velocity[0]):
                min_angular_velocity[0] = state.pole_angle_velocity
        else:
            if abs(state.pole_angle_velocity) < abs(min_angular_velocity[1]):
                min_angular_velocity[1] = state.pole_angle_velocity
            if abs(state.pole_angle_velocity) > abs(max_angular_velocity[1]):
                max_angular_velocity[1] = state.pole_angle_velocity

    steps=episode.total_reward
       
print("Max velocity: ", max_velocity)
print("Min velocity: ", min_velocity)
print("Max angle: ", max_angle)
print("Min angle: ", min_angle)
print("Max angular velocity: ", max_angular_velocity)
print("Min angular velocity: ", min_angular_velocity)
print("steps: ", steps)
print("max pso:",max_position)




"""
Max velocity:  [np.float64(3.223877216522474), np.float64(-3.258557402319852)]
Min velocity:  [np.float64(3.005851789535975e-05), np.float64(-9.606011766644107e-05)]
Max angle:  [np.float64(0.42613279326154063), np.float64(-0.41454811566713234)]
Min angle:  [np.float64(3.7123346650833056e-06), np.float64(-2.8237311640033624e-06)]
Max angular velocity:  [np.float64(6.558890918014729), np.float64(-6.533446246380679)]
Min angular velocity:  [np.float64(8.444263223994497e-06), np.float64(-0.00011147008250506074)]
steps:  7.0
max pso: [np.float64(0.4), np.float64(-0.4)]
"""