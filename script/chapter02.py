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
