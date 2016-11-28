import scipy.io as sio
import numpy as np

fileName='MocapData.mat'

data=sio.loadmat(fileName)['data'][0]
print(data)
X = np.concatenate([np.asarray(frame) for frame in data],0)