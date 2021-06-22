# 경사하강법 1
import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 경사하강법 1
learning_rate = 0.1
gradient = tf.reduce_mean((hypothesis - Y) * X) # cost function을 미분한 값
descent = W - learning_rate * gradient
update = W.assign(descent)

# 경사하강법 2  //// 경사하강법 1과 동일한 동작을 한다.
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# train = optimizer.minimize(cost)
#--------------------------------------------

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

