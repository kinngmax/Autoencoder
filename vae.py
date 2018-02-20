import os
import numpy as np
import sklearn
import cv2
import tensorflow as tf
import math

par_dir = "E:/Tensorflow/notMNIST/notMNIST_small"
path = os.listdir(par_dir)
image_list = []
label=0
label_list = []
batch_size = 64

for folder in path:
    images = os.listdir(par_dir + '/' + folder)
    for image in images:
        if(os.path.getsize(par_dir +'/'+ folder +'/'+ image) > 0):
            img = cv2.imread(par_dir +'/'+ folder +'/'+ image, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_list.append(img)
            label_list.append(label)
        else:
            print('File' + par_dir +'/'+ folder +'/'+ image + 'is empty')
    label += 1


print("Looping done")

image_array = np.array(image_list, dtype=np.float32)
image_array = np.reshape(image_array, [len(image_list), 28, 28, 1])
'''
cv2.imshow('test', image_array[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
label_array = np.array(label_list)

one_hot = np.eye(10)[label_array]

image_data, one_hot = sklearn.utils.shuffle(image_array, one_hot)

print("Data ready. Bon Apetiet!")

image_train, label_train = image_data[0:12800], one_hot[0:12800]
image_test, label_test = image_data[12800:17920], one_hot[12800:17920]

def get_train_image(input):

    batch_images = image_train[(input*batch_size):((input+1)*batch_size)]
    batch_label = label_train[(input*batch_size):((input+1)*batch_size)]
    return batch_images, batch_label


## Variational Autoencoder

latent_dim = 20
h_dim = 500
n_pixel = 28*28
lr = 0.005
epochs = 5

X = tf.placeholder(dtype=tf.float32, shape=[None,n_pixel], name='X')

global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

#Encoder:

def weight_saver():
    np.savetxt("E:/Tensorflow/Autoencoder/checkpoint/We.csv", We.eval(), delimiter=",")

##Layer 1:

We = tf.Variable(tf.random_normal(shape=[n_pixel,h_dim], stddev=0.1, dtype=tf.float32), name='We')
Be = tf.Variable(tf.zeros([h_dim], dtype=tf.float32), name='Be')

h_enc = tf.nn.tanh(tf.matmul(X,We) + Be)

## Layer 2:

W_mu = tf.Variable(tf.random_normal(shape=[h_dim,latent_dim], stddev=0.1, dtype=tf.float32), name='W_mu')
B_mu = tf.Variable(tf.zeros([latent_dim], dtype=tf.float32), name='B_mu')

mu = tf.matmul(h_enc,W_mu) + B_mu

W_sig = tf.Variable(tf.random_normal(shape=[h_dim,latent_dim], stddev=0.1, dtype=tf.float32), name='W_sig')
B_sig = tf.Variable(tf.zeros(shape=[latent_dim], dtype=tf.float32), name='B_sig')

sig = tf.matmul(h_enc,W_sig) + B_sig

eps = tf.random_normal(shape=[1, latent_dim])

z = mu + tf.multiply(eps, tf.exp(0.5*sig))

#Decoder:
##Layer 1

Wd = tf.Variable(tf.random_normal(shape=[latent_dim,h_dim], stddev=0.1, dtype=tf.float32), name='Wd')
Bd = tf.Variable(tf.zeros(shape=[h_dim], dtype=tf.float32), name='Bd')

h_dec = tf.nn.tanh(tf.matmul(z,Wd) + Bd)

Wr = tf.Variable(tf.random_normal(shape=[h_dim, n_pixel], stddev=0.1, dtype=tf.float32), name='Wr')
Br = tf.Variable(tf.zeros(shape=[n_pixel], dtype=tf.float32), name='Br')

recon = tf.nn.sigmoid(tf.matmul(h_dec,Wr) + Br)

# Loss Function:

log_like = tf.reduce_sum((X*tf.log(recon + 1e-9)) + ((1-X)*tf.log(1 - recon + 1e-9)), reduction_indices = 1)
mse = tf.reduce_sum(tf.squared_difference(recon, X), 1)

KL = -0.5*tf.reduce_sum(1.0 + 2.0*sig - tf.square(mu) - tf.exp(2.0*sig), 1)

loss = tf.reduce_mean(log_like - KL)
optimizer = tf.train.AdamOptimizer(lr).minimize(-loss, global_step = global_step)

init = tf.global_variables_initializer()

latent_layer = []


# Training Loop:

with tf.Session() as sess:

    loops = int(image_train.shape[0]/batch_size)
    sess.run(init)

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    
    chkpt = tf.train.get_checkpoint_state(os.path.dirname('E:/Tensorflow/Autoencoder/checkpoint/'))
    if chkpt and tf.gfile.Exists(chkpt.model_checkpoint_path):
        saver = tf.train.import_meta_graph("E:/Tensorflow/Autoencoder/checkpoint/vae-model.chkpt-205.meta")
        saver.restore(sess, chkpt.model_checkpoint_path)
        print("Restored")

    initial_step = int(math.floor(global_step.eval()/loops))

    start = global_step.eval()%200

    for i in range(initial_step, epochs):

        total_loss = 0

        for j in range(start, loops):

            batch,l = get_train_image(j)
            #batch = np.reshape(batch, [batch_size,n_pixel])
            _,b_loss,log,kl,lat_l,imgs = sess.run([optimizer, loss, log_like, KL, z, recon], feed_dict={X:np.reshape(batch,[batch_size,n_pixel])})
            total_loss += b_loss
            latent_layer.append(lat_l)
            print("After {0} : Loss: {1} : Avg: {2}".format(j,b_loss,total_loss/image_train.shape[0]))
            if j % 5 == 0:
                saver.save(sess, "E:/Tensorflow/Autoencoder/checkpoint/vae-model.chkpt", loops+epochs)
                #print("After {0} : {1}".format(j,total_loss))

        print("End of {} epoch".format(i))

    print("End of optimization")
