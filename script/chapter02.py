# 2.2 計算グラフでの演算
import tensorflow as tf
sess = tf.Session()

# Numpy配列を作成する
import numpy as np

x_vals = np.array([1., 3., 5., 7., 9.])
x_data = tf.placeholder(tf.float32)
m_const = tf.constant(3.)

# 演算を定義する
my_product = tf.multiply(x_data, m_const)

# 入力値をループで処理し、入力値ごとに乗算の結果を出力する
for x_val in x_vals:
    print(sess.run(my_product, feed_dict={x_data: x_val}))

# 2.3 入れ子の演算を階層化する
import numpy as np
import tensorflow as tf

sess = tf.Session()

# まず、供給するためのデータと、対応するプレースホルダを作成する
my_array = np.array([[1., 3., 5., 7., 9.],
                    [-2., 0., 2., 4., 6.],
                    [-6., -3., 0., 3., 6.]])

# 入力を2にするために配列を複製
x_vals = np.array([my_array, my_array + 1])
x_data = tf.placeholder(tf.float32, shape=(3, 5))

# 次に行列の乗算と加算に使用する定数を作成する
m1 = tf.constant([[1.], [0.], [-1], [2.], [4.]])
m2 = tf.constant([[2.]])
a1 = tf.constant([[10.]])

# 演算を設定し、それらを計算グラフに追加する
# 行列の乗算 (A[3x5] * m1[5x1] = prod1[3x1])
prod1 = tf.matmul(x_data, m1)
# 行列の乗算 (prod1[3x1] * m2[1x1] = prod2[3x1])
prod2 = tf.matmul(prod1, m2)
# 行列の加算 (prod2[3x1] + a1[1x1])
add1 = tf.add(prod2, a1)

# 最後に計算グラフを通じてデータを供給する
for x_val in x_vals:
    print(sess.run(add1, feed_dict={x_data: x_val}))


# 2.4 複数の層を操作する
import numpy as np
import tensorflow as tf
import os

sess = tf.Session()


# 2D画像の作成
x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape)

# プレースホルダーを作成
x_data = tf.placeholder(tf.float32, shape=x_shape)

# フィルター演算
my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
my_strides = [1, 2, 2, 1]
mov_avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides,
                             padding='SAME', name="Moving_Avg_Window")

# 移動平均ウィンドウの2x2出力を操作するカスタム層を定義する
def custom_layer(input_matrix):
    input_matrix_sqeezed = tf.squeeze(input_matrix)
    A = tf.constant([[2., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    temp1 = tf.matmul(A, input_matrix_sqeezed)
    temp = tf.add(temp1, b)
    return(tf.sigmoid(temp))

# カスタム層を計算グラフに追加
with tf.name_scope('Custom_Layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)

# プレースホルダーに画像を供給
print(sess.run(custom_layer1, feed_dict={x_data: x_val}))

# 2.5 損失関数を実装する
import matplotlib.pyplot as plt
import tensorflow as tf
sess = tf.Session()

x_vals = tf.linspace(-1., 1., 500)
target = tf.constant(0.)

# 回帰
l2_y_vals = tf.square(target - x_vals)
l2_y_out = sess.run(l2_y_vals)

l1_y_vals = tf.abs(target - x_vals)
l1_y_vals = sess.run(l1_y_vals)

delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. +
                             tf.square((target - x_vals)/delta1)) - 1.)
phuber1_y_out = sess.run(phuber1_y_vals)

delta2 = tf.constant(0.25)
phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. +
                             tf.square((target - x_vals)/delta2)) - 1.)
phuber2_y_out = sess.run(phuber2_y_vals)

x_vals = tf.linspace(-3., 5., 500)
target = tf.constant(1.)
targets = tf.fill([500, ], 1.)

# 分類
hinge_y_vals = tf.maximum(0., 1. - tf.multiply(target, x_vals))
hinge_y_out = sess.run(hinge_y_vals)

xentropy_y_vals = - tf.multiply(target, tf.log(x_vals)) - \
                    tf.multiply((1. - target), tf.log(1. - x_vals))
xentropy_y_out = sess.run(xentropy_y_vals)


# 2.6 バックプロパゲーションの実装
import numpy as np
import tensorflow as tf

sess = tf.Session()

x_vals = np.random.normal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[1], dtype=tf.float32)

y_target = tf.placeholder(shape=[1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1]))

my_output = tf.multiply(x_data, A)

loss = tf.square(my_output - y_target)

init = tf.global_variables_initializer()
sess.run(init)

import numpy as np
x_vals = np.concatenate((np.random.normal(-1, 1, 1),
           np.random.normal(3, 1, 1)))
print(x_vals)

# バッチトレーニングと確率的トレーニング
import matplolib as plt
import numpy as np
import tensorflow as tf

sess = tf.Session()

batch_size = 20

x_vals = np.random.nornal(1, 0.1, 100)
y_vals = np.repeat(10., 100)
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
A = tf.Variable(tf.random_normal(shape=[1, 1]))

my_output = tf.matmul(x_data, A)
loss = tf.reduce_mean(tf.square(my_output - y_target))

init = tf.global_variables_initializer()
sess.run(init)

my_opt = tf.train.GradientDescentOptimizer(0.02)
train_step = my_opt.minimize(loss)

loss_batch = []
for i in range(100):
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_vals[rand_index]])
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    if (i + 1)%5==0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)))
        temp_loss = sess.run(loss,
                             feed_dict = {x_data: rand_x, y_target: rand_y})
    print('Loss = ' + str(temp_loss))
    loss_batch.append(temp_loss)

# 分類を行うための要素を組み合わせる
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf

sess = tf.Session()

iris = datasets.load_iris()
binary_target = np.array([1. if x==0 else 0. for x in iris.target])

# iris.data = [(がく片の長さ、がく片の幅、花びらの長さ、花びらの幅)]
iris_2d = np.array([[x[2], x[3]] for x in iris.data])

batch_size = 20

x1_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
x2_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# x1 - (x2 * A + b)
my_mult = tf.matmul(x2_data, A)
my_add = tf.add(my_mult, b)
my_output = tf.subtract(x1_data, my_add)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output,
                                                   labels=y_target)

my_opt = tf.train.GradientDescentOptimizer(0.05)
train_step = my_opt.minimize(xentropy)

init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    rand_index = np.random.choice(len(iris_2d), site=batch_size)
    rand_x = iris_2d[rand_index]
    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    rand_y = np.array([[y] for y in binary_target[rand_index]])
    sess.run(train_step, feed_dict={x1_data: rand_x1,
                                    x2_data: rand_x2,
                                    y_target: rand_y})

    if (i+1)%200==0:
        print('Step #' + str(i + 1) + ' A = ' + str(sess.run(A)) + ', b = ' + \
               str(sess.run(b)))
