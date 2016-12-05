import pickle
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.envs.normalized_env import normalize
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

# auto save config
debug_env=  False
experiment_spec = "test|"
save_policy_every = 1200

# show result config
iter_each_policy = 600
max_path_len = 500

# # test env
# env = normalize(HumanoidEnv(
#             vel_deviation_cost_coeff=0,
#             alive_bonus=0,
#             ctrl_cost_coeff=0,
#             impact_cost_coeff=0,
#             disc=pickle.load(open("model/"+experiment_spec+itr_str+"discriminatior.pickle","rb")),
#             vel_threshold=0.5,
#             vel_bonus=0.1,
#         )
#     )

#temps
exper_num = 0
rewards = []
all_rewards = []

while True:
    try:
        itr_str = str((exper_num+1)*save_policy_every)
        # if normalized_obs:
        #     lis = pickle.load(open("model/"+experiment_spec+itr_str+"env.pickle","rb"))
        #     env.set_state(lis)
        #     print("mean")
        #     print(env._obs_mean)
        #     print("var")
        #     print(env._obs_var)
        env = normalize(HumanoidEnv(
                vel_deviation_cost_coeff=0,
                alive_bonus=0,
                ctrl_cost_coeff=0,
                impact_cost_coeff=0,
                #disc=pickle.load(open("model/"+experiment_spec+itr_str+"discriminator.pickle","rb")),
                vel_threshold=0.4,
                vel_bonus=0.2,
            )
        )
        policy = pickle.load(open("model/"+experiment_spec+itr_str+"policy.pickle","rb"))
        exper_num+=1
        tol_reward = 0
        for i in range(iter_each_policy):
            observation = env.reset()

            env.render()
            sum_reward = 0
            for t in range(max_path_len): 
                if debug_env:
                    action = env.action_space.sample()
                else:
                    action, _ = policy.get_action(observation)

                observation, reward, done, _ = env.step(action)
                print(observation*180/3.14)
                if done:
                    break
                env.render()
                sum_reward+= reward
            rewards.append(sum_reward)
        all_rewards.append(rewards)

    except Exception as e:
        print(e)
        sys.exit(0)

all_rewards = np.array(all_rewards)

# np.array([0.09625816, -0.78859199 , 1.01126844 , 0.84804082 , 0.51415685, -0.05513301,
#   0.11588723 , 0.22389235 , 0.31955742 ,-0.59864492 , 0.54277726  ,0.06720192,
#   0.29375958 , 0.20688923  ,0.03635865, -0.27951822,  0.14636601 ,-0.06866802,
#  -0.49370297  ,0.41969977,  0.27991385, -0.22006313, -1.40829135 , 0.24104596,
#  -0.1661085,   0.8193666 ,  1.11275057,  0.55966961, -0.87384416,  0.32109327])

# # plot
# x = np.arange(0, exper_num*save_policy_every, save_policy_every)
# y = np.mean(all_rewards, axis=1)
# yerr = np.std(all_rewards, axis=1)

# plt.errorbar(x, y, xerr=0.0, yerr=yerr)
# plt.show()