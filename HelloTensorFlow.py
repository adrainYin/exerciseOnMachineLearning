# -*- coding: utf-8 -*-
import tensorflow as tf
from mnist_ import read_data_sets

# 加载MNIST数据
input_data = read_data_sets('MNIST_data', one_hot=True)

# 运行TensorFlow的InteractiveSession
sess = tf.InteractiveSession()

# 构建Softmax 回归模型
# 占位符
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# 变量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

''''' 
#初始化变量 
sess.run(tf.initialize_all_variables()) 

#类别预测与损失函数 
y = tf.nn.softmax(tf.matmul(x,W) + b) 

#损失函数是目标类别和预测类别之间的交叉熵 
cross_entropy = -tf.reduce_sum(y_*tf.log(y)) 

#用梯度下降算法以0.01的学习速率最小化交叉熵 
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) 

#开始训练模型，这里我们让模型循环训练1000次 
for i in range(1000): 
  batch = input_data.train.next_batch(50) 
  train_step.run(feed_dict={x: batch[0], y_: batch[1]}) 

#评估模型 
#检测我们的预测是否真实标签匹配(索引位置一样表示匹配)，输出为一组布尔值 
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) 

#为了确定正确预测项的比例，把布尔值转换成浮点数，然后取平均值 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 

#计算所学习到的模型在测试数据集上面的正确率 
print accuracy.eval(feed_dict={x: input_data.test.images, y_: input_data.test.labels}) 
'''


# 构建一个多层卷积网络
# 权重初始化
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


# 卷积和池化
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


# 第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 训练和评估模型
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = input_data.train.next_batch(50)
  if i % 100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
      x: batch[0], y_: batch[1], keep_prob: 1.0})
    print
    "step %d, training accuracy %g" % (i, train_accuracy)
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print
"test accuracy %g" % accuracy.eval(feed_dict={
  x: input_data.test.images, y_: input_data.test.labels, keep_prob: 1.0})


