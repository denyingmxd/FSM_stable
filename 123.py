import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
m = -4
c = 12
noisiness = 7
num_points = 20000
x = ( torch.rand( num_points, 1 ) - 0.5 ) * 10
y = ( ( torch.rand( num_points, 1 ) - 0.5 ) * noisiness ) + ( x * m + c )
plt.scatter( x.tolist(), y.tolist(), color='red' )
plt.show()

xplusone = torch.cat( ( torch.ones( x.size(0),1 ), x) , 1 )
a=time.time()
R, _ = torch.lstsq( y, xplusone )
R = R[0:xplusone.size(1)]
print(time.time()-a)
yh = xplusone.mm( R )
plt.plot( x.tolist(), yh.tolist() )
plt.scatter( x.tolist(), y.tolist(), color='red' )
plt.show()
