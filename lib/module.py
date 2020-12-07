import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.optimizers import Adam
from lib.dnnlib import (generator, critic)
from lib.utils import (batch_feed, gradient_penalty, plot_tiles_gas)


class cWGAN():

    def __init__(self, seed=333, 
                 input='sdmh', target='sgas',
                 alpha=10, beta=0.04, delta=10,
                 replica_batch_size=4, ncri=1, epochs=300000):

        tf.random.set_seed(seed)
        np.random.seed(seed)

        os.mkdir('checkpoints/')
        os.mkdir('imgs/')

        gpus = len(tf.config.experimental.list_physical_devices('GPU'))
        self.strategy = tf.distribute.get_strategy()
        if gpus > 1:
            print('Used no. of gpus: {}'.format(gpus))
            self.strategy = tf.distribute.MirroredStrategy()

        self.replica_batch_size = replica_batch_size
        self.global_batch_size = replica_batch_size * self.strategy.num_replicas_in_sync
        self.epochs = epochs
        self.ncri = ncri
        

        with self.strategy.scope():

            self.gen = generator()
            self.cri = critic()

            self.g_opt = Adam(learning_rate=1e-5, beta_1=0.0, beta_2=0.9)
            self.c_opt = Adam(learning_rate=1e-5, beta_1=0.0, beta_2=0.9)

        self.bf = batch_feed(datapath_x='/path/to/data/', 
                             datapath_y='/path/to/data/', 

                             datapaths_zoom_x='/path/to/data/', 
                             datapaths_zoom_y='/path/to/data/', 
                            
                             input=input,
                             target=target,
                            
                             batchsize=self.global_batch_size - 1,
                             batchsize_zoom=1,
                             tilesize=512,
                             seed=seed)

        fix_points = [[2, 0, 0], [3, 0, 0], [5, 0, 0], [8, 0, 0]]

        self.x_fix, y_fix = self.bf.get_fix_batch(fix_points=fix_points)
        self.y_fix_d = self.downsampling(y_fix)
        
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.generator_aloss, self.generator_loss_dssim, self.generator_loss_rmsle, self.critic_aloss, self.critic_gploss = self.losses_init()


    def downsampling(self, image, sizes=[512, 256, 128, 64, 32, 16, 8, 4]):
        tensors = [tf.image.resize(image, size=[s, s]) for s in sizes]
        return tensors


    def losses_init(self):
        """
        UNDER CONSTRUCTION...
        """
        with self.strategy.scope():

            def generator_aloss(c_fake):
                return -tf.reduce_mean(c_fake)

            def generator_loss_dssim(t, p):
                filter_sizes = [11, 9, 7, 5, 5, 3, 3, 3]
                dssim = 1.0 - tf.reduce_mean([tf.image.ssim(img1=true, 
                                                            img2=pred, 
                                                            max_val=tf.reduce_max(true, axis=[1, 2, 3]), 
                                                            filter_size=fs) 
                                                            for (true, pred, fs) in zip(t, p, filter_sizes)])
                return dssim

            def generator_loss_rmsle(t, p):
                return tf.reduce_mean([tf.reduce_sum(tf.keras.losses.MSLE(true, pred), axis=[1, 2]) for (true, pred) in zip(t, p)])

            def critic_aloss(c_real, c_fake):
                return tf.reduce_mean(c_fake) - tf.reduce_mean(c_real) # mean or sum?, also axis=?

            def critic_gploss(real, fake):
                return gradient_penalty(f=self.cri, real=real, fake=fake, mode='two-sided')
        
        return(generator_aloss,
               generator_loss_dssim,
               generator_loss_rmsle,
               critic_aloss,
               critic_gploss)


    def train_step_critic(self, inputs):

        x, t = inputs
        x = self.downsampling(x)
        t = self.downsampling(t)

        ncount = 0
        while ncount < self.ncri:
            with tf.GradientTape() as c_tape:
            
                p = self.gen(x[0])
                c_real = self.cri(t + x)
                c_fake = self.cri(p + x)

                c_aloss = self.critic_aloss(c_real=c_real, c_fake=c_fake)
                c_gploss = self.critic_gploss(real=t+x, fake=t+p)
                c_loss = c_aloss + self.delta * c_gploss
            
            c_grad = c_tape.gradient(c_loss, self.cri.trainable_variables)
            self.c_opt.apply_gradients(zip(c_grad, self.cri.trainable_variables))
            ncount += 1

        return c_loss


    def train_step_generator(self, inputs):

        x, t = inputs
        x = self.downsampling(x)
        t = self.downsampling(t)

        with tf.GradientTape() as g_tape:
            
            p = self.gen(x[0])
            c_fake = self.cri(p + x)
            g_aloss = self.generator_aloss(c_fake)
            g_dssim = self.generator_loss_dssim(t=t, p=p)
            g_rmsle = self.generator_loss_rmsle(t=t, p=p)
            g_loss = g_aloss + self.alpha * g_dssim + self.beta * g_rmsle
        
        g_grad = g_tape.gradient(g_loss, self.gen.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grad, self.gen.trainable_variables))

        return g_loss


    # `run` replicates the provided computation and runs it with the distributed input.
    @tf.function
    def distributed_critic_train_step(self, dataset_inputs):
        per_replica_losses = self.strategy.run(self.train_step_critic, args=(dataset_inputs,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    @tf.function
    def distributed_generator_train_step(self, dataset_inputs):
        per_replica_losses = self.strategy.run(self.train_step_generator, args=(dataset_inputs,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


    def train(self):

        f = open('metrics.txt', 'w')
        f.close()

        for epoch in range(self.epochs):

            x, y = self.bf.get_batch()
            dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(self.global_batch_size)
            dist_dataset = self.strategy.experimental_distribute_dataset(dataset)

            for I in dist_dataset:
                generator_loss = self.distributed_generator_train_step(I)
                critic_loss = self.distributed_critic_train_step(I)

            if epoch % 10 == 0:
                self.gen.save('checkpoints/g_{0:06d}.h5'.format(epoch))
                self.cri.save('checkpoints/c_{0:06d}.h5'.format(epoch))

            if epoch % 5 == 0:
                f = open('metrics.txt', 'a')
                f.write('{0:}\t{1:}\t{2:}\n'.format(epoch, generator_loss, critic_loss))
                f.close()

            if epoch % 1 == 0:
                plot_tiles_gas(self.gen, self.x_fix, self.y_fix_d, epoch)

