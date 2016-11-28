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
            disc_window=2,
            disc_joints_dim=10,
            learning_rate=0.005,
            train_threshold=0.25, # train when average_disc_loss > train_threshold
            a_max=1.0,
            a_min=1.0,
            batch_size = 64,
            iter_per_train = 1,
            decent_portion=0.8,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=NL.sigmoid,
            disc_network=None,
    ):
        Serializable.quick_init(self, locals())
        self.batch_size=64
        self.iter_per_train=iter_per_train
        self.disc_window = disc_window
        self.disc_joints_dim = disc_joints_dim
        self.disc_dim = self.disc_window*self.disc_joints_dim
        self.end_iter = int(iteration*decent_portion)
        self.iter_count = 0
        self.learning_rate = learning_rate
        self.train_threshold=train_threshold
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

        disc_reward = disc_network.output_layer
        obs_var = disc_network.input_layer.input_var

        disc_var, = L.get_output([disc_reward])

        self._disc_var = disc_var

        LasagnePowered.__init__(self, [disc_reward])
        self._f_disc = ext.compile_function(
            inputs=[obs_var],
            outputs=[disc_var],
            log_name="f_discriminate_forward",
        )
        
        params = L.get_all_params(disc_network, trainable=True)
        loss = lasagne.objectives.squared_error(disc_var, target_var).mean()
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
        if(len(observation.shape)==1):
            observation = observation.reshape((1, observation.shape[0]))
        disc_ob = self.get_disc_obs(observation)
        assert(disc_ob.shape[1] == self.disc_dim)
        reward = self._f_disc(disc_ob)[0]
        return reward[0][0]

    def train(self, observations):
        '''
        observations: length trj_num list of np.array with shape (trj_length, dim)
        '''
        #print("state len: ", len(observations))
        logger.log("fitting discriminator...")
        loss=[]

        for i in range(self.iter_per_train):
            batch_obs = self.get_batch_obs(observations, self.batch_size)
            batch_mocap = self.get_batch_mocap(self.batch_size)
            disc_obs = self.get_disc_obs(batch_obs)
            disc_mocap = batch_mocap
            X = np.vstack((disc_obs, disc_mocap))
            targets = np.zeros([2*self.batch_size, 1])
            targets[self.batch_size :]=1
            curr_loss = self._f_disc_loss(disc_obs, np.zeros([self.batch_size, 1]))
            print(curr_loss)
            if np.mean(curr_loss) > self.train_threshold:
                self._f_disc_train(X, targets)
                logger.log("fitted!")
            else:
                logger.log("yield training: avg_loss under threshold")
            loss.append(curr_loss)

        avg_disc_loss = np.mean(loss)
        logger.record_tabular("averageDiscriminatorLoss", avg_disc_loss)

    def load_data(self, fileName='MocapData.mat'):
        # TODO: X (n, dim) dim must equals to the disc_obs
        data=sio.loadmat(fileName)['data'][0]
        X = np.concatenate([np.asarray(frame) for frame in data],0)
        # usedDim = np.ones(X.shape[1]).astype('bool')
        # usedDim[39:45] = False
        # usedDim = sio.loadmat('limbMask.mat')['mask'][0].astype(bool)
        self.usedDim = [5, 14, 16, 17, 19, 26, 28, 33, 38, 52]
        #only knee
        #self.usedDim = [26,38]
        usedDim = self.usedDim
        X = X[:,usedDim]
        if (X.shape[1] != self.disc_joints_dim):
            print("\n", X.shape[1], self.disc_joints_dim)
        
        c=180.0/np.pi
        print("mocap_disc")
        print(np.max(X, axis=0)/c)
        print(np.min(X, axis=0)/c)
        return X

    def get_batch_mocap(self, batch_size):
        '''
        return np.array of shape (batch_size, mocap_dim*window)
        '''
        mask = np.random.randint(0, self.data.shape[0]-self.disc_window, size=batch_size)
        temp =[]
        for i in range(self.disc_window):
            temp.append(self.data[mask+i])
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

            # Fill in the data that we have
            s = list(state)
            f = np.zeros(62)
            # #data['lhumerus'] = [c*s[19],c*s[18],c*s[17]]
            # #leftshoulder
            # f[51:54] = [c*s[15],c*s[16], 0]
            # #data['lradius'] = [-c*s[20]]
            # #leftelbow angle
            # f[17] = 0
            # #data['rhumerus'] = [c*s[23],c*s[22],c*s[21]]
            # #rightsholder
            # f[4:7] = [c*s[13],c*s[14],0]
            # #data['rradius'] = [-c*s[24]]
            # #rightelbow angle
            # f[14] = 0
            # #data['rfemur'] = [c*s[13],c*s[14],c*s[15]]
            # #right hip x,y,x rotation
            # f[18:21] = [0, c*s[8], c*s[7]]
            # #data['rtibia'] = [c*s[16]]
            # #right knee angle
            # f[26] = c*s[9]
            # #data['lfemur'] = [c*s[9],c*s[10],c*s[11]]
            # #left hip x,y,z rotation
            # f[27:30] = [0, c*s[11], c*s[10]]
            # #data['ltibia'] = [c*s[12]]
            # #left knee angle
            # f[38] = c*s[12]

            # # ankle mujoco ＋蹬 －收 （－50，50）
            # s[14] #right
            # s[20] #left

            # foot 选了第一个维度
            f[33] = s[14]
            f[16] = s[20]

            # # elbow mujoco -蹬 ＋收 （－90，50）
            # s[24] #right
            # s[27] #left
            f[14]= 90+s[24]
            f[17]= 90+s[27]

            # # knee mujoco ＋蹬 －收 （－2，－160）
            # s[13] #right
            # s[19] #left

            f[26] = -s[13]
            f[38] = -s[19]

            # # hip_y        -蹬 ＋收 （－120， 20）
            # s[12] #right
            # s[18] #left
            
            # femur 选了第二个维度
            f[19]=s[12]+90
            f[28] = s[18]+90

            # # shoulder 2    
            # s[23] #right   -收 ＋伸 （－85，60）
            # s[26] #left    －伸 ＋收 （－60， 85）
            f[5] =s[23]
            f[52] = -s[26]


            frames.append(f)
        
        return np.asarray(frames)[:,self.usedDim]