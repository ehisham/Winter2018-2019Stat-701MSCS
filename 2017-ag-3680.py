from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as dd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


b= dd.DataFrame({"F": [100,200,300,400,500,600,700], "R": [10, 20, 10, 30, 20, 20, 30], "Y": [40, 50, 50, 70, 65, 65, 80]})
result = smf.ols(formula="Y ~ F + R", data=b).fit()
print(result.summary())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(b['F'],
           b['R'], b['Y'],
           c='r', marker='o')
ab, ba = np.meshgrid(b['F'],
                     b['R'])
exog = dd.core.frame.DataFrame({'F':ab.ravel(),
                                'R':ba.ravel()}
                               )
out = result.predict(exog=exog)
ax.plot_surface(ab, ba, out.values.reshape(ab.shape), rstride=1, cstride=1, alpha='0.4', color='None')
ax.set_xlabel("F")
ax.set_ylabel("R")
ax.set_zlabel("Y")
plt.show()
