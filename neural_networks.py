# -*- coding: utf-8 -*-
"""
Created on Sat May 22 17:06:38 2021

@author: Natsy
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

"""img = mnist.train.images[0]
img = img.reshape((28,28))
plt.imshow(img, cmap=cm.Greys)
plt.show()"""

CHECKPOINTS_FOLDER = 'checkpoints'

with tf.variable_scope('input'):
    x = tf.placeholder(tf.float32, [None,784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None,10], name='y-true')
    #courbes dans tensorboard
    img_reshaped = tf.reshape(x, [-1,28,28,1])
    tf.summary.image('input', img_reshaped, max_outputs=10)

with tf.variable_scope('fc1'):
    W = tf.get_variable('weights', [784,10]) #poids
    b = tf.get_variable('bias',[10]) #biais
    y = tf.add(tf.matmul(x,W), b, name='y-pred') #prediction du reseau de neurone

with tf.variable_scope('training'):
    #methode de comparaison de la prédiction et la valeur correcte
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
    #on applique le gradient descendant
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.variable_scope('accuracy'):
    #on verifie si la valeur est correcte
    correct_pred = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
    #on calcule la précision
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

"""entrainnement du réseau de neuronnes"""

#sauvegarde des variables
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_writer = tf.summary.FileWriter('train', sess.graph)

#restaurer les données et entrainer à partir du dernier fichier sauvegardé

"""files = os.listdir(CHECKPOINTS_FOLDER)

if len(files) > 0:
    files.remove('checkpoint')
    files = sorted(files, reverse=True)
    checkpoint = os.path.splitext(files[0])[0]
    saver.restore(sess, os.path.join(CHECKPOINTS_FOLDER, checkpoint))
    print('Model restauré')
else:
    print('Aucun model à restaurer')"""

def feed_dict(is_training):
    if is_training:
        batch_x, batch_y = mnist.train.next_batch(100)
    else:
        batch_x, batch_y = mnist.test.images,mnist.test.labels
    return {x: batch_x, y_ : batch_y}

for i in range(10000):
    if i % 100 == 0:
        #sauvegarde
        #saver.save(sess, os.path.join(CHECKPOINTS_FOLDER, 'model'), global_step=i)
        acc = sess.run(accuracy, feed_dict=feed_dict(True))
        print('étape %d: Précision du training: %f' % (i, acc))
    else:
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
        train_writer.flush()

train_writer.close()

print('Précision Test:', sess.run(accuracy, feed_dict=feed_dict(False)))

