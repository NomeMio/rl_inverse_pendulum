from math import exp
import random

import gymnasium as gym
import time


class State:
    def __init__(self, observation):
        self.position = observation[0]
        self.velocity = observation[2]
        self.pole_angle = observation[1]
        self.pole_angle_velocity = observation[3]


class Action:
    def __init__(self, imprinted_velocity:float):
        self.max_imprinted=3.0
        if imprinted_velocity > self.max_imprinted:
            imprinted_velocity = self.max_imprinted
        if imprinted_velocity < -self.max_imprinted:
            imprinted_velocity = -self.max_imprinted
        
        self.action = [imprinted_velocity]
    def get_action(self):
        return self.action
    def get_velocity(self):
        return self.action[0]
    
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
        interaction= Interaction(State(temp[0]), action, temp[1], temp[2], temp[3], temp[4])
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





class ActorPolicyContinuousSpace:
    def __init__(self):
        self.rho_m_size=2
        self.rho_s_size=2
        self.rho_size=4
        self.rho=[random.uniform(-1, 1) for _ in range(self.rho_size)]
        self.alpha_w=0.05
        self.i=1
        self.alpha_rho=0.05
        self.state=None
        self.discount=0.95
        self.w=[random.uniform(-1, 1) for _ in range(self.rho_size)]
        pass

    def value(self, state:State):
        features=self.get_features(state)
        value=sum([self.w[i]*features[i] for i in range(len(features))])
        return value
    
    def action_value(self, state:State):
        m=self.m(state)
        s=self.s(state)
        action_value=random.gauss(m, s)
        return action_value        
    
    def get_action(self, state:State):
        action_value=self.action_value(state)
        action=Action(action_value)
        return action
    
    def m(self, state:State):
        rho_m=self.get_rho_m()
        features_rho_m=self.get_features_rho_m(state)
        m=sum([rho_m[i]*features_rho_m[i] for i in range(len(rho_m))])
        return m
    def s(self, state:State):
        rho_s=self.get_rho_s()
        features_rho_s=self.get_features_rho_s(state)
        s=sum([rho_s[i]*features_rho_s[i] for i in range(len(rho_s))])    
        s = max(-1.0, min(2.0, s))

        s=exp(s)
        return s

    def new_episode(self,state:State):
        self.state=state
        self.i=1

    def update(self, reward, newstate:State,action:Action):
        gamma=reward+self.discount*self.value(newstate)-self.value(self.state)
        features=self.get_features(self.state)
        for i in range(self.rho_size):
            self.w[i]+=self.alpha_w*gamma*features[i]
        a=action.get_velocity()
        rho_m_factor=(1/self.s(self.state)**2)*(a-self.m(self.state))
        rho_s_factor=((a-self.m(self.state))**2/(self.s(self.state)**2)-1)
        for i in range(self.rho_m_size):
            self.rho[i]+=self.alpha_rho*gamma*self.i*rho_m_factor*self.get_features_rho_m(self.state)[i]
        for i in range(self.rho_s_size):
            self.rho[i+self.rho_m_size]+=self.alpha_rho*gamma*self.i*rho_s_factor*self.get_features_rho_s(self.state)[i]
        self.i=self.i*self.discount
        self.state=newstate

    
    def get_features(self, state:State):
        return [
            state.position ,
            state.velocity ,
            state.pole_angle ,
            state.pole_angle_velocity ,
        ]
    
    def get_features_rho_s(self, state:State):
        return [
            state.velocity,
            state.pole_angle_velocity,
        ]
    def get_features_rho_m(self, state:State):
        return [
            state.position ,
            state.pole_angle ,
        ]
    def get_rho_m(self):
        return [self.rho[0], self.rho[1]]

    def get_rho_s(self):
        return [self.rho[2], self.rho[3]]


