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

# 係数を抽出
[[a1], [a2]] = sess.run(A)
[[b]] = sess.run(b)
slope = -a2/a1
y_intercept = b/a1
x1_vals = [d[1] for d in x_vals]

# 最も適合する直線を取得
best_fit = []
for i in x1_vals:
    best_fit.append(slope*i+y_intercept)

# Setosa かどうかで分割
setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i] == [1]]
setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i] == [1]]
not_setosa_x = [d[1] for i,d in enumerate(x_vals) if y_vals[i] == -1]
not_setosa_y = [d[0] for i,d in enumerate(x_vals) if y_vals[i] == -1]

# データと最も適合する直線をプロット
plt.plot(setosa_x, setosa_y, 'o', label='I. setosa')
plt.plot(not_setosa_x, not_setosa_y, 'x', label='Non-setosa')
plt.plot(x1_vals, best_fit, 'r-', label='Linear Separator', linewidth=3)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

plt.plot(train_accuracy, 'k-', label='Training Accuracy')
plt.plot(test_accuracy, 'r--', label='Test Accuracy')
plt.title('Train and Test Set Accuracies')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# 損失地をプロット
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

# 線形回帰への縮約
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

sess = tf.Session()

iris = datasets.load_iris()
# iris.data = [(がく片の長さ、がく片の幅、花びらの長さ、花びらの幅)]
x_vals = np.array([x[3] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])

# データセットをトレーニングセットとテストセットに分割
train_indices = np.random.choice(len(x_vals), round(len(x_vals)*0.8),\
                                 replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

batch_size = 50

x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[1, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

model_output = tf.add(tf.matmul(x_data, A), b)

# 損失関数を指定
epsilon = tf.constant([0.5])
loss = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs( \
                      tf.subtract(model_output, y_target)), epsilon)))

my_opt = tf.train.GradientDescentOptimizer(0.075)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

train_loss = []
test_loss = []
for i in range(200):
    rand_index = np.random.choice(len(x_vals_train), size=batch_size)
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    temp_train_loss = sess.run(loss, feed_dict={
        x_data: np.transpose([x_vals_train]),
        y_target: np.transpose([y_vals_train])})
    train_loss.append(temp_train_loss)

    temp_test_loss = sess.run(loss, feed_dict={
        x_data: np.transpose([x_vals_test]),
        y_target: np.transpose([y_vals_test])})
    test_loss.append(temp_test_loss)

    if (i+1)%50==0:
        print('------------')
        print('Generation:', str(i))
        print(f'A = {str(sess.run(A))} b = f{str(sess.run(b))}')
        print('Train Loss =', str(temp_train_loss))
        print('Test Loss =', str(temp_test_loss))

# 係数を抽出
[[slope]] = sess.run(A)
[[y_intercept]] = sess.run(b)
[width] = sess.run(epsilon)

# 最も適合する直線を取得
best_fit = []
best_fit_upper = []
best_fit_lower = []
for i in x_vals:
    best_fit.append(slope*i+y_intercept)
    best_fit_upper.append(slope*i+y_intercept+width)
    best_fit_lower.append(slope*i+y_intercept-width)

# 最も適合する直線をプロット
plt.plot(x_vals, y_vals, 'o', label='Data Points')
plt.plot(x_vals, best_fit, 'r-', label='SVM Regression Lin', linewidth=3)
plt.plot(x_vals, best_fit_upper, 'r--', linewidth=2)
plt.plot(x_vals, best_fit_lower, 'r--', linewidth=2)
plt.ylim([0, 10])
plt.legend(loc='lower right')
plt.title('Sepal Length vs Pedal Width')
plt.xlabel('Pedal Width')
plt.ylabel('Sepal Length')
plt.show()

# 損失値をプロット
plt.plot(train_loss, 'k-', label='Train Set Loss')
plt.plot(test_loss, 'r--', label='Test Set Loss')
plt.title('L2 Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('L2 Loss')
plt.legend(loc='upper right')
plt.show()
