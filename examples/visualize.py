import pickle
from rllab.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.envs.normalized_env import normalize
import numpy as np
import matplotlib.pyplot as plt
import sys

# auto save config
debug_env = True
experiment_spec = "model"
save_policy_every = 50

# show result config
iter_each_policy = 1000
max_path_len = 5000

# test env
env = normalize(HumanoidEnv())

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

# # plot
# x = np.arange(0, exper_num*save_policy_every, save_policy_every)
# y = np.mean(all_rewards, axis=1)
# yerr = np.std(all_rewards, axis=1)

# plt.errorbar(x, y, xerr=0.0, yerr=yerr)
# plt.show()