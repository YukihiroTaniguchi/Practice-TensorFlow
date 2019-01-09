# 線形回帰
# 逆行列法

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sess = tf.Session()

# データを作成
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)

# 計画行列Aを作成
x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_vals_column, ones_column))

# 行列bを作成
b = np.transpose(np.matrix(y_vals))

# テンソルを作成
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

# 逆行列法
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
tA_A_inv = tf.matrix_inverse(tA_A)
product = tf.matmul(tA_A_inv, tf.transpose(A_tensor))
solution = tf.matmul(product, b_tensor)

solution_eval = sess.run(solution)

# 解から係数 （傾き、切片)を抽出する
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]
print('slope:', slope)
print('y_intercept', y_intercept)


# 最も適合する直線を取得し、結果をプロット
best_fit = []
for i in x_vals:
    best_fit.append(slope * i + y_intercept)

plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.show()




# 行列分解法を実装する
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

ops.reset_default_graph()

sess = tf.Session()

# データを作成
x_vals = np.linspace(0, 10, 100)
y_vals = x_vals + np.random.normal(0, 1, 100)

# 計画行列Aを作成
x_vals_column = np.transpose(np.matrix(x_vals))
ones_column = np.transpose(np.matrix(np.repeat(1, 100)))
A = np.column_stack((x_vals_column, ones_column))

# 行列bを作成
b = np.transpose(np.matrix(y_vals))

# テンソルを作成
A_tensor = tf.constant(A)
b_tensor = tf.constant(b)

# コレスキー分解
tA_A = tf.matmul(tf.transpose(A_tensor), A_tensor)
L = tf.cholesky(tA_A)

# L * y = t(A) * b を解く
tA_b = tf.matmul(tf.transpose(A_tensor), b)
sol1 = tf.matrix_solve(L, tA_b)

# L' * y = sol1 を解く
sol2 = tf.matrix_solve(tf.transpose(L), sol1)
solution_eval = sess.run(sol2)

# 係数と抽出
slope = solution_eval[0][0]
y_intercept = solution_eval[1][0]
print('slope :', slope)
print('y_intercept :', y_intercept)

# 結果のプロット
best_fit = []
for i in x_vals:
    best_fit.append(i * slope + y_intercept)

plt.plot(x_vals, y_vals, 'o', label='Data')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.show()
