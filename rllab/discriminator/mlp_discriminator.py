import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne.init
import theano.tensor as TT
import theano
import lasagne

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.core.network import MLP
from rllab.misc import ext
from rllab.misc import logger

import scipy.io as sio
import numpy as np

class Mlp_Discriminator(LasagnePowered, Serializable):
    def __init__(
            self,
            iteration,
            disc_window=16,
            disc_joints_dim=20,
            learning_rate=0.005,
            train_threshold=0.25, # train when average_disc_loss > train_threshold
            a_max=1.0,
            a_min=1.0,
            batch_size = 64,
            iter_per_train = 1,
            decent_portion=0.8,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=None,
            downsample_factor=1,
            disc_network=None,
            reg=0.08,
            mocap_framerate=120,
            mujoco_apirate=20,
    ):
        Serializable.quick_init(self, locals())
        self.batch_size=64
        self.iter_per_train=iter_per_train
        self.disc_window = disc_window
        self.downsample_factor = downsample_factor 
        self.disc_joints_dim = disc_joints_dim
        self.disc_window_downsampled = (self.disc_window-1)//self.downsample_factor + 1
        self.disc_dim = self.disc_window_downsampled*self.disc_joints_dim
        self.end_iter = int(iteration*decent_portion)
        self.iter_count = 0
        self.learning_rate = learning_rate
        self.train_threshold=train_threshold
        self.reg =reg
        self.rate_factor=mocap_framerate//mujoco_apirate
        out_dim = 1
        target_var = TT.imatrix('targets')

        # create network
        if disc_network is None:
            disc_network = MLP(
                input_shape=(self.disc_dim,),
                output_dim=out_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
            )

        self._disc_network = disc_network

        disc_score = disc_network.output_layer
        self.disc_score = disc_network.output_layer
        obs_var = disc_network.input_layer.input_var

        disc_var, = L.get_output([disc_score])

        self._disc_var = disc_var

        exp_reward = TT.nnet.sigmoid(disc_var)
        LasagnePowered.__init__(self, [disc_score])
        self._f_disc = ext.compile_function(
            inputs=[obs_var],
            outputs=[exp_reward],
            log_name="f_discriminate_forward",
        )
        
        params = L.get_all_params(disc_network, trainable=True)
        batch_loss = TT.nnet.binary_crossentropy(TT.nnet.sigmoid(disc_var), target_var)
        batch_entropy = self.logit_bernoulli_entropy(disc_var)
        loss = (batch_loss-self.reg * batch_entropy).mean()
        updates = lasagne.updates.adam(loss, params, learning_rate=self.learning_rate)
        
        self._f_disc_train = ext.compile_function(
            inputs=[obs_var, target_var],
            outputs=[loss],
            updates=updates,
            log_name="f_discriminate_train"
        )

        self._f_disc_loss = ext.compile_function(
            inputs=[obs_var, target_var],
            outputs=[loss],
            log_name="f_discriminate_loss"
        )

        self.data = self.load_data()
        self.a = np.linspace(a_min, a_max, self.end_iter)

    def get_reward(self, observation):
        if len(observation.shape)==1:
            observation = observation.reshape((1, observation.shape[0]))

        disc_ob = self.get_disc_obs(observation)
        # print(self.disc_dim)
        # print(disc_ob.shape)
        assert(disc_ob.shape[1] == self.disc_dim)
        reward = self._f_disc(disc_ob)[0]
        return reward[0][0]

    def train(self, observations):
        '''
        observations: length trj_num list of np.array with shape (trj_length, dim)
        '''
        #print("state len: ", len(observations))
        logger.log("fitting discriminator...")
        loss={"obs":[], "mocap":[]}

        for i in range(self.iter_per_train):
            batch_obs = self.get_batch_obs(observations, self.batch_size)
            print(batch_obs[10]/3.14*180)
            batch_mocap = self.get_batch_mocap(self.batch_size)
            disc_obs = self.get_disc_obs(batch_obs)
            disc_mocap = batch_mocap
            print("\n\n\n")
            print(disc_obs[10])
            print("\n\n")
            print(disc_mocap[10])
            print("\n\n\n")
            X = np.vstack((disc_obs, disc_mocap))
            targets = np.zeros([2*self.batch_size, 1])
            targets[self.batch_size :]=1
            obs_loss = self._f_disc_loss(disc_obs, np.zeros([self.batch_size, 1]))
            mocap_loss = self._f_disc_loss(disc_mocap, np.ones([self.batch_size, 1]))
            if np.mean(obs_loss) > self.train_threshold:
                self._f_disc_train(X, targets)
                logger.log("fitted!")
            else:
                logger.log("yield training: avg_loss under threshold")
            loss["obs"].append(obs_loss)
            loss["mocap"].append(mocap_loss)
        avg_disc_loss_obs = np.mean(loss["obs"])
        avg_disc_loss_mocap = np.mean(loss["mocap"])
        logger.record_tabular("averageDiscriminatorLoss_mocap", avg_disc_loss_mocap)
        logger.record_tabular("averageDiscriminatorLoss_obs", avg_disc_loss_obs)

    def load_data(self, fileName='MocapData.mat'):
        # X (n, dim) dim must equals to the disc_obs
        data=sio.loadmat(fileName)['data'][0]
        
        X = np.concatenate([np.asarray(frame) for frame in data],0)
        # onepose = data[5][342]
        # X = np.vstack([onepose,]*1000)

        self.usedDim = [4,5,6,51,52,53,27,28,29,18,19,20,17,14,38,26,15,16,32,33]

        usedDim = self.usedDim
        X = X[:,usedDim]
        if (X.shape[1] != self.disc_joints_dim):
            print("\n", X.shape[1], self.disc_joints_dim)
        #print(X)
        return X

    def get_batch_mocap(self, batch_size):
        '''
        return np.array of shape (batch_size, mocap_dim*window)
        '''
        mask = np.random.randint(0, self.data.shape[0]-self.disc_window*self.rate_factor, size=batch_size)
        temp =[]
        for i in range(self.disc_window_downsampled):
            temp.append(self.data[mask+i*self.downsample_factor*self.rate_factor])
        batch_mocap = np.hstack(temp)
        assert(batch_mocap.shape[0]==batch_size)
        assert(batch_mocap.shape[1]==self.disc_dim)
        #print(batch_mocap[10])
        return batch_mocap

    # def get_disc_mocap(self, mocap_batch):
    #     '''
    #     param mocap_batch np.array of shape (batch_size, mocap_dim*window)
    #     return np.array of ashape (batch_size, disc_dim)
    #     '''
    #     temp = mocap_batch[:, self.usedDim]
    #     return temp

    def inc_iter(self):
        self.iter_count+=1

    def get_a(self):
        if self.iter_count < self.end_iter:
            return self.a[self.iter_count]
        else:
            return self.a[-1]

    def get_batch_obs(self, observations, batch_size):
        '''
        params observations: length trj_num list of np.array with shape (trj_length, dim)
        params batch_size: batch_size of obs
        return a np.array with shape (batch_size, observation_dim)
        '''
        observations = np.vstack(observations)
        ob_dim = observations.shape[1]
        mask = np.random.randint(0, observations.shape[0]-self.disc_window, size=batch_size)
        temp = []
        for i in range(self.disc_window):
            temp.append(observations[mask+i])
        batch_obs = np.hstack(temp)

        assert(batch_obs.shape[0]==batch_size)
        assert(len(batch_obs.shape)==2)
        assert(batch_obs.shape[1]==self.disc_window*ob_dim)

        return batch_obs 


    def get_disc_obs(self, observation):
        """
        param observation nparray with shape (n, window*obs_dim)
        return observation nparray with shape(n, disc_dim)
        """
        temp = [self.convertToMocap(s.reshape((self.disc_window, -1))).reshape(-1) for s in observation]
        return np.asarray(temp)

    def convertToMocap(self, states):

        frames = []
        # print(states.shape)
        c=180.0/np.pi
        # Write each frame
        states=states*c

        for state,frame in zip(states,range(len(states))):

            if frame % self.downsample_factor ==0:

                # Fill in the data that we have
                s = list(state)
                f = np.zeros(62)

                # right humerus
                f[4] = s[17+7]
                f[5] = s[16+7]
                f[6] = s[15+7]

                # left humerus
                f[51] = s[21+7]
                f[52] = s[20+7]
                f[53] = s[19+7]

                # left femur
                f[27] = s[11+7]
                f[28] = s[10+7]
                f[29] = s[9+7]

                # right femur
                f[18] = s[5+7]
                f[19] = s[4+7]
                f[20] = s[3+7]

                # radius
                f[17] = s[22+7]
                f[14] = s[18+7]

                # tibia
                f[38] = s[12+7]
                f[26] = s[6+7]

                # left foot
                f[15] = s[14+7]
                f[16] = s[13+7]

                # right foot 
                f[32] = s[8+7] 
                f[33] = s[7+7]

                frames.append(f)
        
        return np.asarray(frames)[:,self.usedDim]

    def set_all_params(self, params):
        L.set_all_param_values(L.get_all_layers(self.disc_score), params)

    def get_all_params(self):
        return L.get_all_param_values(self.disc_score)

    def logit_bernoulli_entropy(self, disc_var):
        ent = (1.-TT.nnet.sigmoid(disc_var))*disc_var + TT.nnet.softplus(-disc_var)
        return ent