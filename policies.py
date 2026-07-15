import ast
import json
import os

from cart_model import *


class ActorPolicyContinuousSpace:
    def __init__(self,alpha_w:float=0.1, alpha_rho:float=0.1, discount:float=0.85):
        self.rho_m_size=2
        self.rho_s_size=2
        self.rho_size=4
        self.rho=[random.uniform(-1, 1) for _ in range(self.rho_size)]
        self.alpha_w=alpha_w
        self.i=1
        self.alpha_rho=alpha_rho
        self.state=None
        self.discount=discount
        self.w=[random.uniform(-1, 1) for _ in range(self.rho_size)]
        pass
    @staticmethod
    def get_paramters_samples():
        return {
            "alpha_w": (0.01,0.05, 0.005),
            "alpha_rho": (0.01,0.05, 0.005),
            "discount": (0.99, 0.9),
        }
    def get_parameters(self):
        return {
            "alpha_w": self.alpha_w,
            "alpha_rho": self.alpha_rho,
            "discount": self.discount,
        }
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

    def update(self, reward, newstate:State,action:Action, terminated:bool=False):
        bootstrap = 0.0 if terminated else self.value(newstate)
        gamma=reward+self.discount*bootstrap-self.value(self.state)
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


    def save(self, filename="saved_policies/actor_policy_continuous_space.json"):
        with open(filename, "w") as f:
            json.dump({
                "w": self.w,
                "rho": self.rho,
                "alpha_w": self.alpha_w,
                "alpha_rho": self.alpha_rho,
                "discount": self.discount
            }, f)

    @staticmethod
    def load(filename="saved_policies/actor_policy_continuous_space.json"):
        with open(filename, "r") as f:
            data = json.load(f)
        policy = ActorPolicyContinuousSpace()
        policy.w = data["w"]
        policy.rho = data["rho"]
        policy.alpha_w = data["alpha_w"]
        policy.alpha_rho = data["alpha_rho"]
        policy.discount = data["discount"]
        return policy
    


import quantizer

class sarsaWithQuantization:
    
    def getParameters(self):
        return {
            "gamma": self.gamma,
            "alpha": self.alpha,
            "actions_buckets": self.actions_buckets
        }
    def createQTable(self):
        self.q_table = {}
        for velocity in range(self.quantizer.bucketizers["velocity"].n_buckets):
            for pole_angle in range(self.quantizer.bucketizers["pole_angle"].n_buckets):
                for pole_angle_velocity in range(self.quantizer.bucketizers["pole_angle_velocity"].n_buckets):
                        state_key = ( velocity, pole_angle, pole_angle_velocity)
                        self.q_table[state_key] = [0 for _ in range(self.actions_buckets)]
    
    def getActionFromIndex(self, action_index:int):
        action_value=self.action_quantizer.bucket_center(action_index)
        action=Action(action_value)
        return action
    
    def get_action(self, state:State,training=False):

        if training:
            epsilon=max(0.05, 0.5*(1/(1+self.episodesTaken/2000)))
        else:
            epsilon=0
        action_class=self.egreedy_action(state, epsilon=epsilon)
        return action_class

    def new_episode(self):
        self.episodesTaken+=1

    def current_alpha(self):
        return max(0.1*self.alpha, self.alpha/(1+self.episodesTaken/5000))
    
    def getActionIndex(self, action:Action):
        return self.action_quantizer.discretize(action.get_velocity())

    def getQValue(self, state:State, action_index:int):
        quantized_state = self.quantizer.discretize(state)
        quantized_state = (quantized_state[1], quantized_state[2], quantized_state[3])  # Exclude position
        return self.q_table[quantized_state][action_index]
    
    def setQValue(self, state:State, action_index:int, value:float):
        quantized_state = self.quantizer.discretize(state)
        quantized_state = (quantized_state[1], quantized_state[2], quantized_state[3])  # Exclude position
        self.q_table[quantized_state][action_index] = value
    
    def egreedy_action(self, state:State, epsilon=0.05):
        quantized_state = self.quantizer.discretize(state)
        quantized_state = (quantized_state[1], quantized_state[2], quantized_state[3])  # Exclude position
        if random.random() < epsilon:
            action_index = random.randint(0, self.actions_buckets - 1)
        else:
            q_values = self.q_table[quantized_state]
            max_q = max(q_values)
            best_indices = [idx for idx, q in enumerate(q_values) if q == max_q]
            action_index = random.choice(best_indices)
        return self.getActionFromIndex(action_index)

    def __init__(self,actions_buckets=10,gamma=0.9, alpha=0.01, action_quantizer_factory=None):
        self.gamma=gamma
        self.stepTaken=0
        self.episodesTaken=0
        self.alpha=alpha
        self.actions_buckets=actions_buckets
        self.quantizer = quantizer.StateQuantizer.load("saved_policies/state_quantizer.json")
        self.action_quantizer = self._load_action_quantizer(actions_buckets, default_factory=action_quantizer_factory)
        self.createQTable()

    @staticmethod
    def _load_action_quantizer(actions_buckets, filename="saved_policies/action_quantizer.json", default_factory=None):
        try:
            loaded = quantizer.ActionQuantizer.load(filename)
            if loaded.n_buckets == actions_buckets:
                return loaded
        except FileNotFoundError:
            pass
        if default_factory is not None:
            return default_factory(actions_buckets)
        return quantizer.ActionQuantizer(-3.0, 3.0, actions_buckets)

    @staticmethod
    def get_paramters_samples():
        return {
            "actions_buckets": (7, 15, 4),
            "gamma": (0.79, 0.99, 0.05),
            "alpha": (0.001, 0.201, 0.05),
        }

    @staticmethod
    def get_action_bucket_sizes():
        """All grid values covered by the actions_buckets range in
        get_paramters_samples(), e.g. (7, 15, 4) -> [7, 11, 15]. Used to
        compare bucket sizes against each other, each with its own full
        random search over the remaining hyperparameters."""
        low, high, step = sarsaWithQuantization.get_paramters_samples()["actions_buckets"]
        n_steps = round((high - low) / step)
        return [int(round(low + k * step)) for k in range(n_steps + 1)]

    def update(self, state:State, action:Action, reward, next_state:State, next_action:Action, terminated:bool=False):
        self.stepTaken+=1
        action_index=self.getActionIndex(action)
        if terminated:
            bootstrap=0.0
        else:
            next_action_index=self.getActionIndex(next_action)
            bootstrap=self.gamma*self.getQValue(next_state,next_action_index)
        update_value=self.current_alpha()*(reward+bootstrap-self.getQValue(state,action_index))
        self.setQValue(state,action_index,self.getQValue(state,action_index)+update_value)

    def save(self, dirname="saved_policies/sarsa_with_quantization"):
        os.makedirs(dirname, exist_ok=True)
        with open(os.path.join(dirname, "policy.json"), "w") as f:
            json.dump({
                "q_table": {str(k): v for k, v in self.q_table.items()},
                "gamma": self.gamma,
                "alpha": self.alpha,
                "actions_buckets": self.actions_buckets
            }, f)
        self.quantizer.save(os.path.join(dirname, "state_quantizer.json"))
        self.action_quantizer.save(os.path.join(dirname, "action_quantizer.json"))

    @staticmethod
    def load(dirname="saved_policies/sarsa_with_quantization"):
        with open(os.path.join(dirname, "policy.json"), "r") as f:
            data = json.load(f)
        policy = sarsaWithQuantization.__new__(sarsaWithQuantization)
        policy.gamma = data["gamma"]
        policy.alpha = data["alpha"]
        policy.actions_buckets = data["actions_buckets"]
        policy.stepTaken = 0
        policy.episodesTaken = 0
        policy.quantizer = quantizer.StateQuantizer.load(os.path.join(dirname, "state_quantizer.json"))
        policy.action_quantizer = quantizer.ActionQuantizer.load(os.path.join(dirname, "action_quantizer.json"))
        policy.q_table = {ast.literal_eval(k): v for k, v in data["q_table"].items()}
        return policy