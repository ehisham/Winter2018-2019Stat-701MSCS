import numpy as np
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd


y = [5,7,15,17,9,11]
q = [0,0,10,10,20,20]
w = [0,0,100,100,400,400]
x = np.column_stack((q,w))
x = sm.add_constant(x, prepend=True)
res = sm.OLS(y,x).fit()
print res.params
print res.bse
print res.summary()

fig = plt.figure()
axis = fig.add_subplot(111, projection='3d')
axis.scatter(['q'],['w'],['Y'], c='r', marker='o')
xx, yy = np.meshgrid(['q'],['w'])
exog = pd.core.frame.DataFrame({'q':xx.ravel(),'w':yy.ravel()})
out = res.predict(exog=exog)
axis.plot_surface(xx, yy, out.values.reshape(xx.shape), rstride=1, cstride=1, alpha='0.2', color='None')
axis.set_xlabel("q")
axis.set_ylabel("w")
axis.set_zlabel("y")
plt.show()
