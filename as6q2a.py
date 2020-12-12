#Q(ii)a final
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.figsize'] = (10.0, 5.0)
df = pd.read_csv("week6.csv", comment = '#')
m=3
Xtrain = np.array(df.iloc[:, 0]).reshape(-1, 1)
ytrain = np.array(df.iloc[:, 1]).reshape(-1, 1)
Xtest = np.linspace(-3.0, 3.0, num = 1000).reshape(-1, 1)

def gaussian_kernel(distances):
    weights = np.exp(0*(distances**2))
    return weights/np.sum(weights)

def gaussian_kernel1(distances):
    weights = np.exp(-1 * (distances ** 2))
    return weights/np.sum(weights)

def gaussian_kernel5(distances):
    weights = np.exp(-5 * (distances ** 2))
    return weights/np.sum(weights)

def gaussian_kernel10(distances):
    weights = np.exp(-10 * (distances ** 2))
    return weights/np.sum(weights)

def gaussian_kernel25(distances):
    weights = np.exp(-25 * (distances ** 2))
    return weights/np.sum(weights)

from sklearn.neighbors import KNeighborsRegressor 
model = KNeighborsRegressor(n_neighbors = 998, weights = gaussian_kernel).fit(Xtrain, ytrain)
ypred = model.predict(Xtest)
model2 = KNeighborsRegressor(n_neighbors = 998, weights = gaussian_kernel1).fit(Xtrain, ytrain)
ypred2 = model2.predict(Xtest)
model3 = KNeighborsRegressor(n_neighbors = 998, weights = gaussian_kernel5).fit(Xtrain, ytrain)
ypred3 = model3.predict(Xtest)
model4 = KNeighborsRegressor(n_neighbors = 998, weights = gaussian_kernel10).fit(Xtrain, ytrain)
ypred4 = model4.predict(Xtest)
model5 = KNeighborsRegressor(n_neighbors = 998, weights = gaussian_kernel25).fit(Xtrain, ytrain)
ypred5 = model5.predict(Xtest)

plt.scatter(Xtrain, ytrain, color = 'orange', label = 'train')
plt.plot(Xtest, ypred, color = 'grey', label = 'γ = 0')
plt.plot(Xtest, ypred2, color = 'red', label = 'γ = 1')
plt.plot(Xtest, ypred3, color = 'blue', label = 'γ = 5')
plt.plot(Xtest, ypred4, color = 'green', label = 'γ = 10')
plt.plot(Xtest, ypred5, color = 'purple', label = 'γ = 25')
plt.xlabel("input x"); plt.ylabel("output y")
plt.legend()
plt.show()