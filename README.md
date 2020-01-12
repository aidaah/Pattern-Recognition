# Pattern-Recognition
Clusters and Classification Boundaries

```ruby
import math
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

## Generating data:

X_A = np.random.multivariate_normal(np.array([5, 10]), np.array([[8, 4], [4, 40]]), 100)
X_B = np.random.multivariate_normal(np.array([15, 10]), np.array([[8, 0], [0, 8]]), 200)
X_C = np.random.multivariate_normal(np.array([10, 5]), np.array([[10, -5], [-5, 20]]), 150)


## Maximum likelihood:

def ML(X):
    N = X.shape[0]
    mean_X = X.mean(axis=0)
    norm_X = X - np.expand_dims(mean_X, axis=0)
    cov_X = 1 / N * norm_X.transpose().dot(norm_X)
    return mean_X, cov_X


def ML_predict(X, mean_class, cov_class):
    norm_X = X - np.expand_dims(mean_class, axis=0)
    p_class = np.diag(
        np.exp(-1 / (2 * math.pi * math.sqrt(np.linalg.det(cov_class))) *
               norm_X.dot(np.linalg.solve(cov_class, norm_X.transpose())))
    )
    return p_class


mean_A, cov_A = ML(X_A)
print('mean A: \n' + str(mean_A))
print('cov A: \n' + str(cov_A))

mean_B, cov_B = ML(X_B)
print('mean B: \n' + str(mean_B))
print('cov B: \n' + str(cov_B))

mean_C, cov_C = ML(X_C)
print('mean C: \n' + str(mean_C))
print('cov C: \n' + str(cov_C))

## Plot the decision boundary

n = 501
ygrid, xgrid = np.meshgrid(np.linspace(-10, 30, n), np.linspace(-5, 25, n))
X = np.vstack([xgrid.ravel(), ygrid.ravel()]).transpose()

p_A = np.zeros((n, n))
p_B = np.zeros((n, n))
p_C = np.zeros((n, n))
for m in range(n):
    p_A[m, :] = ML_predict(X[m * n:(m + 1) * n, :], mean_A, cov_A)
    p_B[m, :] = ML_predict(X[m * n:(m + 1) * n, :], mean_B, cov_B)
    p_C[m, :] = ML_predict(X[m * n:(m + 1) * n, :], mean_C, cov_C)
class_table = np.zeros((n, n)).astype(int)
class_table[(p_A >= p_B) & (p_A >= p_C)] = 0
class_table[(p_B >= p_A) & (p_B >= p_C)] = 1
class_table[(p_C >= p_A) & (p_C >= p_B)] = 2

fig = plt.figure()
plt.plot(X_A[:, 0], X_A[:, 1], '.')
plt.plot(X_B[:, 0], X_B[:, 1], 'x')
plt.plot(X_C[:, 0], X_C[:, 1], 'o')
plt.contourf(xgrid, ygrid, class_table)
plt.show()
fig.savefig('ML_classifier.png')

## Maximum A Posterior (to complete)
```
