import scipy.io as sio
import numpy as np

fileName='MocapData.mat'

data=sio.loadmat(fileName)['data'][0]

i=0

for path in data:
    if path.shape[0]==343:
        print(i)
    i=i+1

a = [  7.12502000e+00,  -1.95534000e+01,   1.35020000e+01,  -1.06930000e+01,
  -6.37887000e+01,  -3.43173000e+01 , -4.71111000e+01 ,  5.99575000e-01,
   1.32881000e+00,  -4.89607000e+00 , -2.41466000e+01 ,  5.75060000e+00,
  -1.88034000e+00  , 9.89992000e-01   ,9.93192000e+00  ,-1.36748000e+01,
  -6.18019000e-01  , 6.52280000e+01 , -2.32433000e+01   ,6.73071000e+00,
   2.50009000e+01 , -1.32314000e+00 , -1.20217000e+00 ,  2.10433000e+00,
   7.12502000e+00  , 2.34686000e+01 ,  2.77761000e+01,   1.69276000e+01,
  -2.85303000e+00 , -2.25749000e+01,  -7.23079000e-15 , -1.27222000e-14,
  -4.35066000e+00 , -1.51615000e+01 ,  8.01561000e-01 ,  4.27075000e+01,
  -7.23079000e-15,  -1.27222000e-14 ,  2.77778000e+01 , 1.10237000e+01,
   1.75020000e+01  , 2.94538000e+01 , -4.48244000e+00 ,  6.81263000e+00,
  -1.59521000e+00 , -1.15801000e+01,   7.38567000e-01,  -1.03767000e+00,
   2.56386000e+01 ,  6.77091000e+00 , -1.63697000e+01,  -3.20358000e+01,
  -3.35783000e+00  , 9.14085000e+01 , -2.57351000e+01 ,  1.27292000e+01,
   2.22766000e+00  , 4.94526000e-01  ,-1.32385000e+00 ,  1.78225000e+00,
  -2.40255000e+00  , 2.42913000e+00],

a = data[5][342]

print(a)

for i in range(len(a)):
    if a[i]==-51.9812  :
        print(i)

b = data[5]
max_b = np.max(b, axis=0)
min_b = np.min(b, axis=0)

print(max_b[16], min_b[16])

X = np.concatenate([np.asarray(frame) for frame in data],0)


print(np.log(0.1))




