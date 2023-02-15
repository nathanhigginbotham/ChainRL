
############################################## Algorithms Dictionary ##############################################

"""
Here

"""


from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy as SACPolicy

from stable_baselines3 import A2C
from stable_baselines3.a2c.policies import MlpPolicy as A2CPolicy

from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy as PPOPolicy

from sb3_contrib import ARS
from sb3_contrib.ars.policies import ARSPolicy

from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import RecurrentActorCriticPolicy

from sb3_contrib import TQC
from sb3_contrib.tqc.policies import MlpPolicy as TQCPolicy

from sb3_contrib import TRPO
from sb3_contrib.trpo.policies import MlpPolicy as TRPOPolicy

algorithm_dictionary = {'SAC'   : [SAC, SACPolicy],
             'A2C'   : [A2C, A2CPolicy],
             'PPO' : [PPO, PPOPolicy],
             'ARS' : [ARS, ARSPolicy],  
             'RecurrentPPO'  : [RecurrentPPO, RecurrentActorCriticPolicy], 
             'TQC'  : [TQC, TQCPolicy],
             'TRPO'  : [TRPO, TRPOPolicy]
            }


############################################## Environments Dictionary ##############################################

"""
Here we have a dictionary with the environments that we want to use. We can add new environments or variations of the existing ones here.


The structure so far is as follows:
    'key' : ['ENVIRONMENT_STRING']


"""




import or_gym

environment_dictionary = {'IM0'   : ["InvManagement-v0"],
             'IM1'   : ["InvManagement-v1"],
             'NET0' : ["NetworkManagement-v0"],
             'NET1' : ["NetworkManagement-v1"] 
            }


############################################## Parameters Dictionary ##############################################

"""
Bit of a block here.

There are a few ways we can go.


Creating a triple nested dicionary:
DICTIONARY -> ENVIRONMENT -> ALGORITHM -> HYPERPARAMETERS

Multiple dictionaries but with the same structure starting with environment at the top level:
ENVIRONMENT -> ALGORITHM -> HYPERPARAMETERS

Multiple dictionaries but with the same structure starting with algorithm at the top level:
ALGORITHM -> ENVIRONMENT -> HYPERPARAMETERS


"""



parameters_dictionary = {
    "IM1": {
        "ingredients": {
            "water": 50,
            "coffee": 18,
        },
        "cost": 1.5,
    },
    "latte": {
        "ingredients": {
            "water": 200,
            "milk": 150,
            "coffee": 24,
        },
        "cost": 2.5,
    },
    "cappuccino": {
        "ingredients": {
            "water": 250,
            "milk": 100,
            "coffee": 24,
        },
        "cost": 3.0,
    }
}