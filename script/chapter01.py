# 1.3 テンソルを設定する
# 固定テンソルを作成
import tensorflow as tf

row_dim, col_dim = 10, 10

zero_tsr = tf.zeros([row_dim, col_dim])

ones_tsr = tf.ones([row_dim, col_dim])

filled_tsr = tf.fill([row_dim, col_dim], 42)

constant_tsr = tf.constant([1, 2, 3])

# 同じような形状のテンソルを複数作成
zero_similar = tf.zeors_like(constant_tsr)

ones_similar = tf.ones_like(constant_tsr)


# シーケンステンソル
linear_tsr = tf.linspace(start=0, stop=1, num=3)

integer_seq_tsr = tf.range(start=6, limit=15, delta=3)

# ランダムテンソル
randunif_tsr = tf.random_uniform([row_dim, col_dim], minval=0, maxval=1)

randnorm_tsr = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)

randnorm_tsr = tf.truncated_normal([row_dim, col_dim], mean=0.0, stddev=1.0)

shuffled_output = tf.random_shuffle(input_tensor)

cropped_output = tf.random_crop(input_tensor, crop_size)

# テンソルをVariable()でラップすることで、対応する変数を作成する
my_var = tf.Variable(tf.zeros([row_dim, col_dim]))

# 1.4 プレースホルダと変数を使用する
# 変数を作成して初期化する
my_var = tf.Variable(tf.zeros([2, 3]))
sess = tf.Session()
initialize_op = tf.global_variables_initializer()
sess.run(initialize_op)

# プレースホルダを作成し、データを供給する
sess = tf.Session()
x = tf.placeholder(tf.float32, shape=[2, 2])
y = tf.identity(x)
x_vals = np.random.rand(2, 2)
sess.run(y, feed_dict={x:x_vals})

# 変数の初期化
initializer_op = tf.global_variables_initializer()

sess = tf.Session()
first_var = tf.Variable(tf.zeros([2, 3]))
sess.run(first_var.initializer)
second_var = tf.Variable(tf.zeros_like(first_var))
# first_varに依存
sess.run(second_var.initializer)


# 行列を操作する
import numpy as np
import tensorflow as tf
sess = tf.Session()

identity_matrix = tf.diag([1.0, 1.0, 1.0])
A = tf.truncated_normal([2, 3])
B = tf.fill([2, 3], 5.0)
C = tf.random_uniform([3, 2])
D = tf.convert_to_tensor(np.array(
    [[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
print(sess.run(identity_matrix))
print(sess.run(A))
print(sess.run(B))
print(sess.run(C))
print(sess.run(D))

# 行列の加算減算
print(sess.run(A+B))
print(sess.run(A-B))
# 行列の乗算
print(sess.run(tf.matmul(B, identity_matrix)))
# 転置
print(sess.run(tf.transpose(C)))
# 行列式
print(sess.run(tf.matrix_determinant(D)))
# 逆行列
print(sess.run(tf.matrix_inverse(D)))
# コレスキー分解
print(sess.run(tf.cholesky(identity_matrix)))
eigenvalues, eigenvectors = sess.run(tf.self_adjoint_eig(D))
print(eigenvalues)
print(eigenvectors)
