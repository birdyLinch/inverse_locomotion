from rllab.envs.base import Step
from .mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import autoargs


class SimpleHumanoidEnv(MujocoEnv, Serializable):

    FILE = 'simple_humanoid.xml'

    @autoargs.arg('vel_deviation_cost_coeff', type=float,
                  help='cost coefficient for velocity deviation')
    @autoargs.arg('alive_bonus', type=float,
                  help='bonus reward for being alive')
    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for control inputs')
    @autoargs.arg('impact_cost_coeff', type=float,
                  help='cost coefficient for impact')
    def __init__(
            self,
            vel_deviation_cost_coeff=1e-2,
            alive_bonus=0.2,
            ctrl_cost_coeff=1e-3,
            impact_cost_coeff=1e-5,
            disc=None,
            vel_threshold=0.4,
            vel_bonus=0.2,
            *args, **kwargs):
        self.vel_deviation_cost_coeff = vel_deviation_cost_coeff
        self.alive_bonus = alive_bonus
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.impact_cost_coeff = impact_cost_coeff
        self.disc=disc
        self.states = []
        self.vel_threshold = vel_threshold
        self.vel_bonus=vel_bonus
        super(SimpleHumanoidEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def get_current_obs(self):
        data = self.model.data
        # print(np.asarray(data.qpos.flat))
        return np.concatenate([
            data.qpos.flat,
            data.qvel.flat,
            # np.clip(data.cfrc_ext, -1, 1).flat,
            self.get_body_com("torso").flat,
        ])

    def _get_com(self):
        data = self.model.data
        mass = self.model.body_mass
        xpos = data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()

        # for disc
        if self.disc!=None:
            self.states.append(next_obs)

        alive_bonus = self.alive_bonus
        data = self.model.data

        comvel = self.get_body_comvel("torso")

        
        lin_vel_reward = comvel[0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = .5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        impact_cost = .5 * self.impact_cost_coeff * np.sum(
            np.square(np.clip(data.cfrc_ext, -1, 1)))
        vel_deviation_cost = 0.5 * self.vel_deviation_cost_coeff * np.sum(
            np.square(comvel[1:]))

        disc_reward=0.0
        if self.disc!=None:
            disc_states=np.hstack(self.states[-self.disc.disc_window:])
            disc_score = self.disc.get_reward(disc_states)
            
            disc_reward = self.disc.get_a() * - np.log(1 - disc_score)             
            self.disc.inc_iter()

            # clip the line_vel_reward
            if (lin_vel_reward>self.vel_threshold):
                lin_vel_reward=self.vel_bonus
            elif lin_vel_reward< -self.vel_threshold:
                lin_vel_reward= -self.vel_bonus

        reward = lin_vel_reward + alive_bonus - ctrl_cost - \
            impact_cost - vel_deviation_cost + disc_reward




        done =data.qpos[2] < 0.8 or data.qpos[2] > 2.0

        return Step(next_obs, reward, done)

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))

    @overrides
    def reset(self, init_state=None):
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        curr_obs = self.get_current_obs()
        
        if self.disc!=None:
            self.states=[]
            zero_obs = np.zeros_like(curr_obs)
            for i in range(self.disc.disc_window-1):
                self.states.append(zero_obs)
            self.states.append(curr_obs)

        return curr_obs

    def set_discriminator_params(self, params):
        self.disc.set_all_params(params)

    def get_discriminator_params(self, params):
        return self.disc.get_all_params()
