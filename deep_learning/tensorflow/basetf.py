import tensorflow as tf
import numpy as np
rank_0_tensor = tf.constant(value=[4],dtype=tf.float32)
print(rank_0_tensor)
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)

# 定义张量a和b
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) 

print(tf.add(a, b), "\n") # 计算张量的和
print(tf.multiply(a, b), "\n") # 计算张量的元素乘法
print(tf.matmul(a, b), "\n") # 计算乘法
print(tf.reduce_sum(a),
      tf.reduce_mean(a),
      tf.reduce_min(a),
      tf.reduce_max(a),
      tf.argmax(a),
      tf.argmin(a))