


#### Initial Framework for a More Streamlined Approach?

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


alg_dictionary = {'sac'   : [SAC, SACPolicy, "SAC"],
             'a2c'   : [A2C, A2CPolicy, "A2C"],
             'ppo' : [PPO, PPOPolicy, "PPO"],
             'ars' : [ARS, ARSPolicy, "ARS"],
             'RecurrentPPO'  : [RecurrentPPO, RecurrentActorCriticPolicy, "RecurrentPPO"], 
             'TQC'  : [TQC, TQCPolicy, "TQC"],
             'TRPO'  : [TRPO, TRPOPolicy, "TRPO"]
            }


value_list = alg_dictionary['sac']
print('Values of key "sac" are:')
print(value_list)