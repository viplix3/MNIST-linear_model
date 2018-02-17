"""
Network Structure
Feed-forward 
Rough Network Structure:-
Input layer >> Weights >> Hidden layer 1 >> (Activation function) >> Weights >> Hidden layer 2 >> (Activation function) 
>> Weights >> Output Layer


Compare output to intended outout >> Cost function (cross-entropy)
Optimizer function (optimize) >> minimize cost (AdamOptimizer) 

Back-propagation

Feed forward + Back prop = epoch
"""

# Importing Stuff
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
from IPython.display import clear_output
from termcolor import colored

# Downloading data to our disk
from tensorflow.python.training import saver

mnist = input_data.read_data_sets('DataSet', one_hot=True)


# Defining the network structure
h1_nodes_1 = 500 # Nodes in hidden layer 1
h1_nodes_2 = 500 # Nodes in hidden layer 2
h1_nodes_3 = 500 # Nodes in hidden layer 3

num_classes = 10
batch_size = 100

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)

x = tf.placeholder('float', [None, img_size_flat])
y = tf.placeholder('float', [None, num_classes])


def neural_network_model(data):

    hidden_layer_1 = {'Weights': tf.Variable(tf.random_normal([img_size_flat, h1_nodes_1])),
                      'biases': tf.Variable(tf.random_normal([h1_nodes_1]))}

    hidden_layer_2 = {'Weights': tf.Variable(tf.random_normal([h1_nodes_1, h1_nodes_2])),
                      'biases': tf.Variable(tf.random_normal([h1_nodes_2]))}

    hidden_layer_3 = {'Weights': tf.Variable(tf.random_normal([h1_nodes_2, h1_nodes_3])),
                      'biases': tf.Variable(tf.random_normal([h1_nodes_3]))}

    output_layer = {'Weights': tf.Variable(tf.random_normal([h1_nodes_3, num_classes])),
                    'biases': tf.Variable(tf.random_normal([num_classes]))}

    layer_1 = tf.add(tf.matmul(data, hidden_layer_1['Weights']), hidden_layer_1['biases'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, hidden_layer_2['Weights']), hidden_layer_2['biases'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, hidden_layer_3['Weights']), hidden_layer_3['biases'])
    layer_3 = tf.nn.relu(layer_3)

    output_layer = tf.matmul(layer_3, output_layer['Weights']) + output_layer['biases']

    return output_layer


def train_neural_network(X):

    prediction = neural_network_model(data=X)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # Default learning rate to AdamOptimizer is 0.001 so we are not providing any as we'll be using that only
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # epochs is one whole cycle of feedforward and backprop though out the whole dataset
    num_epochs = 15

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for epochs in range(num_epochs):

            start_time = time.time()
            epoch_loss = 0
            for w in range(int(mnist.train.num_examples/batch_size)):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # 1 represents the axis
                accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))
                acc = accuracy.eval({x: mnist.train.images, y: mnist.train.labels})
                clear_output(wait=True)
                print('Epoch:', epochs, '|', 'Loss:', colored(c, 'red'), '|', 'Train Accuracy', colored(acc, 'green'), '|', w, 'out of',
                      int(mnist.train.num_examples/batch_size), ' iterations done')

            end_time = time.time()
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # 1 represents the axis
            accuracy = tf.reduce_mean(tf.cast(correct, 'float32'))
            print('Epoch', epochs, 'completed out of', num_epochs, '|', 'Loss:', colored(epoch_loss, 'red'), '|', 'Accuracy',
                  colored(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}), 'green'), '|', 'Time taken:', end_time-start_time)
        save_path = saver.save(sess, "trained_model.ckpt")
        print("Model saved in file: %s" % save_path)


train_neural_network(x)


