import tensorflow as tf

print("Hello world!")

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)  # implies tf.float32
# print(node1, node2)
sess = tf.Session()
print(sess.run([node1, node2]))  # creates a Session object and then invokes its run method to run enough of the
# computational graph to evaluate node1 and node2

node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

a = tf.placeholder(dtype=tf.float32)
b = tf.placeholder(dtype=tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
# The preceding three lines are a bit like a function or a lambda in which we define two input parameters (a and b)
# and then an operation on them.

print(sess.run(adder_node, {a: 3, b: 4.5}))  # 7.5
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))  # [3. 7.]

# lets make it more complex by adding another operation to the computational graph
add_and_triple = adder_node * 3
print("Triple:")
print(sess.run(add_and_triple, {a: 3, b: 4.5}))  # 22.5

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
# To make the model trainable, we need to be able to modify the graph to get new outputs with the same input.
# Variables allow us to add trainable parameters to a graph. They are constructed with a type and initial value.

init = tf.global_variables_initializer()
sess.run(init)
# Constants are initialized when you call tf.constant, and their value can never change. To initialize all the
# variables in a TensorFlow program, you must explicitly call the above special operation special operation.
print("Linear model:")
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))  # [ 0.          0.30000001  0.60000002  0.90000004]
# since x is a placeholder, we evaluate linear_model for several values of x simultaneously

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print("Loss:")
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
# A loss function measures how far apart the current model is from the provided data. We'll use a standard loss model
#  for linear regression, which sums the squares of the deltas between the current model and the provided data.
# linear_model - y creates a vector where each element is the corresponding example's error delta. We call tf.square
# to square that error. Then, we sum all the squared errors to create a single scalar that abstracts the error of all
# examples using tf.reduce_sum.

"""Tf.train API"""

optimizer = tf.train.GradientDescentOptimizer(0.01)  # learning rate for GD optimizer
train = optimizer.minimize(loss)

sess.run(init)  # reset values to incorrect defaults.
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

# print(sess.run([W, b]))

# evaluate training accuracy
x_train, y_train = [1, 2, 3, 4], [0, -1, -2, -3]  # same values as earlier, just rewritten
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
