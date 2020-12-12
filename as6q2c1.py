#Q(ii)c(i) use cross-validation kernalised ridge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.figsize'] = (10.0, 5.0)
df = pd.read_csv("week6.csv", comment = '#')
Xtrain = np.array(df.iloc[:, 0]).reshape(-1, 1)
ytrain = np.array(df.iloc[:, 1]).reshape(-1, 1)

mean_error = []; std_error = []
c_range = [0.1, 1, 10, 1000]
gk_range = [0, 1, 5, 10, 25]

for gk in gk_range:
    from sklearn.kernel_ridge import KernelRidge
    model = KernelRidge(alpha=1.0/(2*0.1), kernel='rbf', gamma=gk)
    temp = []
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 5)
    for train, test in kf.split(Xtrain):
        model.fit(Xtrain[train], ytrain[train])
        ypred = model.predict(Xtrain[test])
        from sklearn.metrics import mean_squared_error
        temp.append(mean_squared_error(ytrain[test], ypred))
    mean_error.append(np.array(temp).std())
    std_error.append(np.array(temp).std())
plt.errorbar(gk_range, mean_error, yerr = std_error, label = 'Error Bar')
plt.xlabel('gk'); plt.ylabel('Mean Square Error')
plt.title("C = 0.1"); plt.legend()
plt.show()

mean_error = []; std_error = []
for gk in gk_range:
    from sklearn.kernel_ridge import KernelRidge
    model = KernelRidge(alpha=1.0/(2*1), kernel='rbf', gamma=gk)
    temp = []
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 5)
    for train, test in kf.split(Xtrain):
        model.fit(Xtrain[train], ytrain[train])
        ypred = model.predict(Xtrain[test])
        from sklearn.metrics import mean_squared_error
        temp.append(mean_squared_error(ytrain[test], ypred))
    mean_error.append(np.array(temp).std())
    std_error.append(np.array(temp).std())
plt.errorbar(gk_range, mean_error, yerr = std_error, label = 'Error Bar')
plt.xlabel('gk'); plt.ylabel('Mean Square Error')
plt.title("C = 1"); plt.legend()
plt.show()

mean_error = []; std_error = []
for gk in gk_range:
    from sklearn.kernel_ridge import KernelRidge
    model = KernelRidge(alpha=1.0/(2*10), kernel='rbf', gamma=gk)
    temp = []
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 5)
    for train, test in kf.split(Xtrain):
        model.fit(Xtrain[train], ytrain[train])
        ypred = model.predict(Xtrain[test])
        from sklearn.metrics import mean_squared_error
        temp.append(mean_squared_error(ytrain[test], ypred))
    mean_error.append(np.array(temp).std())
    std_error.append(np.array(temp).std())
plt.errorbar(gk_range, mean_error, yerr = std_error, label = 'Error Bar')
plt.xlabel('gk'); plt.ylabel('Mean Square Error')
plt.title("C = 10"); plt.legend()
plt.show()

mean_error = []; std_error = []
for gk in gk_range:
    from sklearn.kernel_ridge import KernelRidge
    model = KernelRidge(alpha=1.0/(2*1000), kernel='rbf', gamma=gk)
    temp = []
    from sklearn.model_selection import KFold
    kf = KFold(n_splits = 5)
    for train, test in kf.split(Xtrain):
        model.fit(Xtrain[train], ytrain[train])
        ypred = model.predict(Xtrain[test])
        from sklearn.metrics import mean_squared_error
        temp.append(mean_squared_error(ytrain[test], ypred))
    mean_error.append(np.array(temp).std())
    std_error.append(np.array(temp).std())
import matplotlib.pyplot as plt
plt.errorbar(gk_range, mean_error, yerr = std_error, label = 'Error Bar')
plt.xlabel('gk'); plt.ylabel('Mean Square Error')
plt.title("C = 1000"); plt.legend()
plt.show()