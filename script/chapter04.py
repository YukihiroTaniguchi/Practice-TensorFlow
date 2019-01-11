# 線形SVMを操作する
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()

iris = datasets.load_iris()

# iris.data = [(がく片の長さ、がく片の幅、花びらの長さ、花びらの幅)]
x_vals = np.array([[x[0], x[3]] for x in iris.data])

# Setosa である場合は1、そうでない場合は-1
y_vals = np.array([1 if y==0 else -1 for y in iris.target])

train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8), \
                                 replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

# バッチサイズ、プレースホルダ、モデルの変数を設定する

batch_size = 100

x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[2, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

model_output = tf.subtract(tf.matmul(x_data, A), b)

# L2 ノルム関数を定義。squareで各要素の2乗をとり、reduce_sumで各要素の総和をとる
l2_norm = tf.reduce_sum(tf.square(A))

# L2 正則化パラメータを指定
alpha = tf.constant([0.01])

# 損失関数のマージン項
classification_term = \
    tf.reduce_mean(tf.maximum(0., tf.subtract(1.,
                                  tf.multiply(model_output, y_target))))

# 損失関数の第1項と第2項を加算
loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))

# 予測関数を指定
prediction = tf.sign(model_output)
# 正解率を算出する正解関数を指定
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), \
                                  tf.float32))

# 最適化関数を作成し、モデルの変数を初期化する
my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

# 勾配、切片、損失値などを求める処理を繰り返すトレーニングループ
loss_vec = []
train_accuracy = []
test_accuracy = []
for i in range(500):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = x_vals_train[rand_index]
    rand_y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)

    train_acc_temp = sess.run(accuracy, feed_dict={
        x_data: x_vals_train,
        y_target: np.transpose([y_vals_train])})
    train_accuracy.append(train_acc_temp)

    test_acc_temp = sess.run(accuracy, feed_dict={
        x_data: x_vals_test,
        y_target: np.transpose([y_vals_test])})
    test_accuracy.append(test_acc_temp)

    if (i+1)%100==0:
        print(f'Step #{str(i+1)} A = {str(sess.run(A))}, b = {str(sess.run(b))}')
        print('Loss =', str(temp_loss))
