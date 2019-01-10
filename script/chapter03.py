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


# 3.4 TenforFLow での線形回帰の実装パターン
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops

ops.reset_default_graph()

sess = tf.Session()

# Iris データセット
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# 学習率とバッチサイズを設定
learning_rate = 0.05
batch_size = 25

# プレースホルダを初期化
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 変数を作成
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# y = A_x + b
model_output =tf.add(tf.matmul(x_data, A), b)

# L2損失関数を指定
loss = tf.reduce_mean(tf.square(y_target - model_output))

# 変数を初期化
init = tf.global_variables_initializer()
sess.run(init)

# 最適化関数を指定
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

loss_vec = []
for i in range(100):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    if (i+1)%25==0:
        print('Setep #', str(i+1), 'A =', str(sess.run(A)), \
                                   'b =', str(sess.run(b)))
        print('Loss =', str(temp_loss))

# 係数を抽出
[slope] = sess.run(A)
[y_intercept] = sess.run(b)

# 最も適合する直線を取得
best_fit = []
for i in x_vals:
    best_fit.append(slope*i+y_intercept)

# 1つ目のグラフ
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')

# 2つ目のグラフ
plt.plot(loss_vec, 'k-')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')

# 3.5 線形回帰の損失関数を理解する
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()

# Irirs データセットのデータをロード
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# バッチサイズ、学習率、繰り返しの回数を設定
batch_size = 25
learning_rate = 0.1 # 学習率が0.4を超えると収束しなくなる
iterations = 50

# プレースホルダ、変数、モデルの演算を定義
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
model_output = tf.add(tf.matmul(x_data, A), b)

# 損失関数をL1損失関数に変更する
loss_l1  = tf.reduce_mean(tf.abs(y_target - model_output))

# 変数を初期化
init = tf.global_variables_initializer()
sess.run(init)

# 最適化関数を决定
my_opt_l1 = tf.train.GradientDescentOptimizer(learning_rate)
train_step_l1 = my_opt_l1.minimize(loss_l1)

# トレーニングループを開始
loss_vec_l1 = []
for i in range(iterations):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step_l1, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss_l1 = sess.run(loss_l1,\
                         feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec_l1.append(temp_loss_l1)
    if (i+1)%25==0:
        print('Step #', str(i+1) + 'A =', str(sess.run(A)),\
                                   'b =', str(sess.run(b)))


# 3.6 デミング回帰を実装する
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()

# Irirs データセットのデータをロード
iris = datasets.load_iris()
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# バッチサイズ、学習率、繰り返しの回数を設定
batch_size = 25
learning_rate = 0.1 # 学習率が0.4を超えると収束しなくなる
iterations = 50

# プレースホルダ、変数、モデルの演算を定義
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
model_output = tf.add(tf.matmul(x_data, A), b)

# 損失関数を分子と分母に分けて定義
demming_numerator = tf.abs(tf.subtract(y_target, \
                                       tf.add(tf.matmul(x_data, A), b)))

demming_denominator = tf.sqrt(tf.add(tf.square(A), 1))

loss = tf.reduce_mean(tf.truediv(demming_numerator, demming_denominator))

# 変数の初期化
init = tf.global_variables_initializer()
sess.run(init)

# 最適化関数を指定
my_opt = tf.train.GradientDescentOptimizer(0.25)
train_step = my_opt.minimize(loss)

# トレーニング
loss_vec = []
for i in range(1500):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    if (i+1)%100==0:
        print('Setep #', str(i+1), 'A =', str(sess.run(A)), \
                                   'b =', str(sess.run(b)))
        print('Loss =', str(temp_loss))

# 係数を抽出
[slope] = sess.run(A)
[y_intercept] = sess.run(b)

# 最も適合する直線を取得
best_fit = []
for i in x_vals:
    best_fit.append(slope*i+y_intercept)

# 1つ目のグラフ
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='Best fit line', linewidth=3)
plt.legend(loc='upper left')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
