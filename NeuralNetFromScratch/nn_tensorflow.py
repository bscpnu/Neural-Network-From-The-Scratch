import tensorflow as tf

## author: Imam Mustafa Kamal
## email: imamkamal52@gmail.com

# Model Parameters
W1 = tf.Variable([0.2],name='weight1')
W2 = tf.Variable([0.1],name='weight2')
W3 = tf.Variable([0.3],name='weight3')
W4 = tf.Variable([0.0],name='weight4')
W5 = tf.Variable([0.4],name='weight5')
W6 = tf.Variable([1.0],name='weight6')
W7 = tf.Variable([2.0],name='weight7')
W8 = tf.Variable([0.7],name='weight8')
W9 = tf.Variable([0.8],name='weight9')

b1 = tf.Variable([1.0],name='bias1')
b2 = tf.Variable([0.4],name='bias2')
b3 = tf.Variable([2.0],name='bias3')
b4 = tf.Variable([3.0],name='bias3')

# Model inputs # training data
X1 = tf.placeholder(tf.float32, name='X1')
X2 = tf.placeholder(tf.float32, name='X2')
# y
Y = tf.placeholder(tf.float32, name='Y')

# Model definitation
H1 = X1*W1 + X2*W4 + b1
H2 = X1*W2 + X2*W5 + b2
H3 = X1*W3 + X2*W6 + b3

O1 = H1*W7 + H2*W8 + H3*W9 + b4

# loss function mse
loss = tf.losses.mean_squared_error(O1,Y)

# training op
train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# historical data
x1 = [1.1,2.0,3.5,4.8]
x2 = [4.1,6.1,4.9,7.2]
y = [2.0,3.4,4.2,5.1]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for train_step in range(2000):
        sess.run(train, {X1: x1, X2: x2, Y: y})

    weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9, bias1, bias2, bias3, bias4, loss = sess.run(
        [W1, W2, W3, W4, W5, W6, W7, W8, W9, b1, b2, b3, b4, loss], {X1: x1, X2: x2, Y: y})

    print("W1: %s W2: %s W3: %s W4: %s W5: %s W6: %s W7: %s W8: %s W9: %s b1: %s b2: %s b3: %s loss: %s" % (
        weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9, b1, b2, b3, loss))

    # for prediction
    x1 = 4.8
    x2 = 7.2
    H1 = x1 * W1 + x2 * W4 + b1
    H2 = x1 * W2 + x2 * W5 + b2
    H3 = x1 * W3 + x2 * W6 + b3

    output = H1 * W7 + H2 * W8 + H3 * W9 + b4
    print("prediction X1=1.1,  X2=4.1, Y=", sess.run(output))
    # end prediction