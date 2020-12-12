#Q(ii)c(ii) use cross-validation kernalised ridge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.figsize'] = (10.0, 5.0)
df = pd.read_csv("week6.csv", comment = '#')
Xtrain = np.array(df.iloc[:, 0]).reshape(-1, 1)
ytrain = np.array(df.iloc[:, 1]).reshape(-1, 1)
Xtest = np.linspace(-3.0, 3.0, num = 1000).reshape(-1, 1)

from sklearn.kernel_ridge import KernelRidge
model = KernelRidge(alpha=1.0/(2*10), kernel='rbf', gamma=5).fit(Xtrain, ytrain)
ypred = model.predict(Xtest)
plt.scatter(Xtrain, ytrain, color = 'orange', label = 'train')
plt.plot(Xtest, ypred, color = 'black', label = 'prediction')
plt.xlabel("input x"); plt.ylabel("output y")
plt.legend()
plt.show()
