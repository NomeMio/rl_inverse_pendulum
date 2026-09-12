import gymnasium as gym
import time


class State:
    def __init__(self, observation):
        self.states=["position", "velocity", "pole_angle", "pole_angle_velocity"]
        self.position = observation[0]
        self.velocity = observation[2]
        self.pole_angle = observation[1]
        self.pole_angle_velocity = observation[3]


class Action:
    def __init__(self, imprinted_velocity:float):
        self.max_imprinted=3.0
        self.raw_velocity=imprinted_velocity
        clipped_velocity = imprinted_velocity
        if clipped_velocity > self.max_imprinted:
            clipped_velocity = self.max_imprinted
        if clipped_velocity < -self.max_imprinted:
            clipped_velocity = -self.max_imprinted

        self.action = [clipped_velocity]
    def get_action(self):
        return self.action
    def get_velocity(self):
        return self.action[0]
    def get_raw_velocity(self):
        return self.raw_velocity
    
class Interaction:
    def __init__(self, state:State, action:Action, reward:float, terminated:bool, truncated:bool, info:dict):
        self.state = state
        self.action = action
        self.reward = reward
        self.terminated = terminated
        self.truncated = truncated
        self.info = info

    def __str__(self):
        return f"State: position={self.state.position}, velocity={self.state.velocity}, pole_angle={self.state.pole_angle}, pole_angle_velocity={self.state.pole_angle_velocity}, Action: imprinted_velocity={self.action.get_velocity()}, Reward: {self.reward}, Terminated: {self.terminated}, Truncated: {self.truncated}, Info: {self.info}"

class Cart_model:
    def __init__(self,human:True, env_name:str = "InvertedPendulum-v5", reset_noise:float = 0.1):
        if human:
            self.env = gym.make(env_name, reset_noise_scale=0.1, render_mode="human")
        else:
            self.env = gym.make(env_name, reset_noise_scale=reset_noise)

    def reset(self):
        observation, info = self.env.reset()
        return State(observation), info
    
    def sample_random_action(self):
        action = self.env.action_space.sample()
        action = Action(action[0])
        return action
    
    def do_action(self, action:Action):
        temp=self.env.step(action.get_action())

        max_position = 0.4
        if temp[0][0] > max_position:
            temp[0][0] = max_position
        elif temp[0][0] < -max_position:
            temp[0][0] = -max_position  

        
        state=State(temp[0])
        
        
        if abs(state.position)==max_position:
            interaction= Interaction(state, action, -1, True, True, {"SpaceLimitTruncation": True})
        else:    
            interaction= Interaction(state, action, temp[1], temp[2], temp[3], temp[4])
        return interaction
    
    def close(self):
        self.env.close()



class Episode:
    def __init__(self):
        self.interactions=[]
        self.total_reward=0.0
        self.terminated=False
        self.truncated=False
        self.info=None
        self.starting_state=None

    def run(self, cart_model:Cart_model, policy=None, max_steps:int = 1000):
        state, info = cart_model.reset()
        self.starting_state=state
        for step in range(max_steps):
            if policy is None:
                action=cart_model.sample_random_action()
            else:
                action=policy.get_action(state)
            interaction=cart_model.do_action(action)
            self.interactions.append(interaction)
            self.total_reward+=interaction.reward
            state=interaction.state
            if interaction.terminated or interaction.truncated:
                self.terminated=interaction.terminated
                self.truncated=interaction.truncated
                self.info=interaction.info
                break

    def one_step(self, cart_model:Cart_model, policy=None):
        if len(self.interactions)==0:
            state, info = cart_model.reset()
            self.starting_state=state
        else:
            state=self.interactions[-1].state
        if policy is None:
            action=cart_model.sample_random_action()
        else:
            action=policy.get_action(state)
        interaction=cart_model.do_action(action)
        self.interactions.append(interaction)
        self.total_reward+=interaction.reward
        state=interaction.state
        if interaction.terminated or interaction.truncated:
            self.terminated=interaction.terminated
            self.truncated=interaction.truncated
            self.info=interaction.info
        return state,action,interaction.reward
    
    def runSlow(self, cart_model:Cart_model, policy=None, max_steps:int = 1000, delay:float = 0.1):
        state, info = cart_model.reset()
        self.starting_state=state
        for step in range(max_steps):
            if policy is None:
                action=cart_model.sample_random_action()
            else:
                action=policy.get_action(state)
            interaction=cart_model.do_action(action)
            self.interactions.append(interaction)
            self.total_reward+=interaction.reward
            state=interaction.state
            time.sleep(delay)
            if interaction.terminated or interaction.truncated:
                self.terminated=interaction.terminated
                self.truncated=interaction.truncated
                self.info=interaction.info
                break
