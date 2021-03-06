from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.formula.api as an
from statsmodels.stats.anova import anova_lm



a = pd.DataFrame({"Fertilizer": [100,200,300,400,500,600,700], "Rainfall": [10, 20, 10, 30, 20, 20, 30], "Yield": [40, 50, 50, 70, 65, 65, 80]})
result = an.ols(formula="Yield ~ Fertilizer + Rainfall", data=a).fit()
print result.params
print(anova_lm(result))
print(result.summary())

fig = plt.figure()
axis = fig.add_subplot(111, projection='3d')
axis.scatter(a['Fertilizer'], a['Rainfall'], a['Yield'], c='r', marker='o')
xx, yy = np.meshgrid(a['Fertilizer'],a['Rainfall'])
exog = pd.core.frame.DataFrame({'Fertilizer':xx.ravel(),'Rainfall':yy.ravel()})
out = result.predict(exog=exog)
axis.plot_surface(xx, yy, out.values.reshape(xx.shape), rstride=1, cstride=1, alpha='0.2', color='None')
axis.set_xlabel("Fertilizer")
axis.set_ylabel("Rainfall")
axis.set_zlabel("Yield")
plt.show()

