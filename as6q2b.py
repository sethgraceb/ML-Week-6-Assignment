#Q(ii)b final
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
C = 0.1
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=0).fit(Xtrain, ytrain)
ypredK1 = modelK.predict(Xtest)
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=1).fit(Xtrain, ytrain)
ypredK2 = modelK.predict(Xtest)
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=5).fit(Xtrain, ytrain)
ypredK3 = modelK.predict(Xtest)
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=10).fit(Xtrain, ytrain)
ypredK4 = modelK.predict(Xtest)
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=25).fit(Xtrain, ytrain)
ypredK5 = modelK.predict(Xtest)

plt.scatter(Xtrain, ytrain, color = 'orange', label = 'train')
plt.plot(Xtest, ypredK1, color='grey', label = 'γ = 0')
plt.plot(Xtest, ypredK2, color='red', label = 'γ = 1')
plt.plot(Xtest, ypredK3, color='blue', label = 'γ = 5')
plt.plot(Xtest, ypredK4, color='green', label = 'γ = 10')
plt.plot(Xtest, ypredK5, color='purple', label = 'γ = 25')
plt.xlabel("input x"); plt.ylabel("output y")
plt.title("C = 0.1"); plt.legend()
plt.show()

C = 1
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=0).fit(Xtrain, ytrain)
ypredK1 = modelK.predict(Xtest)
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=1).fit(Xtrain, ytrain)
ypredK2 = modelK.predict(Xtest)
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=5).fit(Xtrain, ytrain)
ypredK3 = modelK.predict(Xtest)
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=10).fit(Xtrain, ytrain)
ypredK4 = modelK.predict(Xtest)
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=25).fit(Xtrain, ytrain)
ypredK5 = modelK.predict(Xtest)

plt.scatter(Xtrain, ytrain, color = 'orange', label = 'train')
plt.plot(Xtest, ypredK1, color='grey', label = 'γ = 0')
plt.plot(Xtest, ypredK2, color='red', label = 'γ = 1')
plt.plot(Xtest, ypredK3, color='blue', label = 'γ = 5')
plt.plot(Xtest, ypredK4, color='green', label = 'γ = 10')
plt.plot(Xtest, ypredK5, color='purple', label = 'γ = 25')
plt.xlabel("input x"); plt.ylabel("output y")
plt.title("C = 1"); plt.legend()
plt.show()

C = 10
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=0).fit(Xtrain, ytrain)
ypredK1 = modelK.predict(Xtest)
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=1).fit(Xtrain, ytrain)
ypredK2 = modelK.predict(Xtest)
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=5).fit(Xtrain, ytrain)
ypredK3 = modelK.predict(Xtest)
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=10).fit(Xtrain, ytrain)
ypredK4 = modelK.predict(Xtest)
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=25).fit(Xtrain, ytrain)
ypredK5 = modelK.predict(Xtest)

plt.scatter(Xtrain, ytrain, color = 'orange', label = 'train')
plt.plot(Xtest, ypredK1, color='grey', label = 'γ = 0')
plt.plot(Xtest, ypredK2, color='red', label = 'γ = 1')
plt.plot(Xtest, ypredK3, color='blue', label = 'γ = 5')
plt.plot(Xtest, ypredK4, color='green', label = 'γ = 10')
plt.plot(Xtest, ypredK5, color='purple', label = 'γ = 25')
plt.xlabel("input x"); plt.ylabel("output y")
plt.title("C = 10"); plt.legend()
plt.show()

C = 1000
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=0).fit(Xtrain, ytrain)
ypredK1 = modelK.predict(Xtest)
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=1).fit(Xtrain, ytrain)
ypredK2 = modelK.predict(Xtest)
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=5).fit(Xtrain, ytrain)
ypredK3 = modelK.predict(Xtest)
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=10).fit(Xtrain, ytrain)
ypredK4 = modelK.predict(Xtest)
modelK = KernelRidge(alpha=1.0/(2*C), kernel='rbf', gamma=25).fit(Xtrain, ytrain)
ypredK5 = modelK.predict(Xtest)

plt.scatter(Xtrain, ytrain, color = 'orange', label = 'train')
plt.plot(Xtest, ypredK1, color='grey', label = 'γ = 0')
plt.plot(Xtest, ypredK2, color='red', label = 'γ = 1')
plt.plot(Xtest, ypredK3, color='blue', label = 'γ = 5')
plt.plot(Xtest, ypredK4, color='green', label = 'γ = 10')
plt.plot(Xtest, ypredK5, color='purple', label = 'γ = 25')
plt.xlabel("input x"); plt.ylabel("output y")
plt.title("C = 1000"); plt.legend()
plt.show()