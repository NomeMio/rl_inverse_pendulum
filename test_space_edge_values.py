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

"""
Obtained values:
Max velocity:  [3.3020846646483535, -3.717943610046488]
Min velocity:  [2.959612574404691e-06, -1.166139481734163e-05]
Max angle:  [0.4348863505502279, -0.41033682635665636]
Min angle:  [2.7573416705381493e-06, -1.1367399891694385e-05]
Max angular velocity:  [6.9075086085739095, -6.648303495892968]
Min angular velocity:  [0.00018444125457239835, -0.00039769929497079914]
"""