import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('pdf')

# ---------------------------------------------------------------------------------------

class batch_feed():

    def __init__(self, seed, 

                 input, target, tilesize,
                 
                 batchsize,
                 datapath_x, datapath_y, 
                 
                 batchsize_zoom,
                 datapaths_zoom_x, datapaths_zoom_y):

        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.tilesize  = tilesize
        self.batchsize = batchsize
        self.batchsize_zoom = batchsize_zoom

        self.x_maps, self.y_maps = [], []
        self.x_maps_zoom, self.y_maps_zoom = [], []

        for axis in ['x', 'y']:
            for i in range(10):
                x = np.load(datapath_x + '{}_{}_{}.npy'.format(input, axis, i))
                y = np.load(datapath_y + '{}_{}_{}.npy'.format(target, axis, i))

                x = np.pad(x, mode='wrap', pad_width=[(0, 256), (0, 256)])
                y = np.pad(y, mode='wrap', pad_width=[(0, 256), (0, 256)])

                self.x_maps.append(x)
                self.y_maps.append(y)

        self.x_maps = np.asarray(self.x_maps)
        self.y_maps = np.asarray(self.y_maps)
        self.maximum = self.x_maps[0].shape[0] - int(self.tilesize)

        if self.batchsize_zoom > 0:
            for (dpz_x, dpz_y) in zip(datapaths_zoom_x, datapaths_zoom_y):
                for num in range(10):
                    x = np.load(dpz_x + '{}_{}.npy'.format(input, num))
                    y = np.load(dpz_y + '{}_{}.npy'.format(target, num))

                    self.x_maps_zoom.append(x)
                    self.y_maps_zoom.append(y)

        self.x_maps_zoom = np.asarray(self.x_maps_zoom)
        self.y_maps_zoom = np.asarray(self.y_maps_zoom)


    def randomize_tile(self, x, y):

        nmir = np.random.randint(0, 2)
        nrot = np.random.randint(0, 4)

        if nmir:
            x = np.fliplr(x)
            y = np.fliplr(y)

        x = np.rot90(x, k=nrot)
        y = np.rot90(y, k=nrot)
        return (x, y)


    def get_batch(self):

        x_batch, y_batch = [], []

        for bs in range(self.batchsize):
            idx = np.random.randint(0, self.x_maps.shape[0] - 1)
            ptx, pty = np.random.randint(low=0, high=self.maximum, size=(2,))

            slc = np.s_[idx,
                        ptx : ptx + self.tilesize,
                        pty : pty + self.tilesize]

            x = self.x_maps[slc]
            y = self.y_maps[slc]

            x, y = self.randomize_tile(x, y)

            x_batch.append(x)
            y_batch.append(y)

        if self.batchsize_zoom > 0:
            for bs in range(self.batchsize_zoom):
                idx = np.random.randint(0, self.x_maps_zoom.shape[0] - 1)
                maximum = self.x_maps_zoom[idx].shape[0] - int(self.tilesize)

                print(self.x_maps_zoom[idx].shape[0])
                
                if maximum <= int(self.tilesize):
                    ptx, pty = 0, 0
                else:
                    ptx, pty = np.random.randint(low=0, high=maximum, size=(2,))

                slc = np.s_[ptx: ptx + self.tilesize,
                            pty: pty + self.tilesize]
                
                x = self.x_maps_zoom[idx][slc]
                y = self.y_maps_zoom[idx][slc]

                x, y = self.randomize_tile(x, y)

                x_batch.append(x)
                y_batch.append(y)

        x_batch = np.asarray(x_batch)[..., np.newaxis]
        y_batch = np.asarray(y_batch)[..., np.newaxis]
        
        return (x_batch, y_batch)


    def get_fix_batch(self, fix_points):

        x_batch, y_batch = [], []

        for pt in fix_points:
            idx, ptx, pty = pt
            slc = np.s_[idx,
                        ptx : ptx + self.tilesize,
                        pty : pty + self.tilesize]

            x = self.x_maps[slc]
            y = self.y_maps[slc]

            x_batch.append(x)
            y_batch.append(y)

        x_batch = np.asarray(x_batch)[..., np.newaxis]
        y_batch = np.asarray(y_batch)[..., np.newaxis]

        return (x_batch, y_batch)            

# ---------------------------------------------------------------------------------------

def gradient_penalty(f, real, fake=None, mode='two-sided'):

    def _interpolate_wgan_gp(a, b=None):
        shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
        epsilon = tf.random.uniform(shape=shape, minval=0., maxval=1.)

        if b is None:
            beta = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            b = a + 0.5 * tf.math.reduce_std(a) * beta

        inter = a + epsilon * (b - a)
        return inter


    if fake is None:
        x = [_interpolate_wgan_gp(a) for a in real]
    else:
        x = [_interpolate_wgan_gp(a, b) for (a, b) in zip(real, fake)]

    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = f(x, training=False)
    grad    = tape.gradient(pred, x)
    slopes  = [tf.sqrt(1e-8 + tf.reduce_sum(tf.square(g), axis=[1, 2, 3])) for g in grad]


    if mode == 'one-sided':
        gp = tf.reduce_mean((tf.maximum(0., tf.convert_to_tensor(slopes) - 1)) ** 2)
    elif mode=='two-sided':
        gp = tf.reduce_mean((tf.convert_to_tensor(slopes) - 1) ** 2)

    return gp

# ---------------------------------------------------------------------------------------

def plot_tiles_gas(gen, x, true_images, epoch, noise_fix=None):

    if noise_fix is None:
        fake_images = gen.predict(x)
    else:
        fake_images = gen.predict([x] + noise_fix)

    f, ax = plt.subplots(8, 8, figsize=(12, 8))
    f.subplots_adjust(hspace=0.1, wspace=0.1)

    cmap = 'bone'
    norm = matplotlib.colors.LogNorm(vmin=1, vmax=100)

    for a in ax.flat:
        a.axis('off')

    for i in range(8):
        for j in range(8):
            if j % 2 == 0:
                field = true_images[i][j//2][...,0]
                ax[i, j].imshow(field * 100, cmap=cmap, norm=norm)
            else:
                field = fake_images[i][j//2][...,0]
                ax[i, j].imshow(field *100, cmap=cmap, norm=norm)

    plt.savefig('imgs/ms_{0:06d}.png'.format(epoch), dpi=250)
    plt.close()


def plot_tiles_HI(gen, x, true_images, epoch, noise_fix=None):

    fake_images = gen.predict(x) # or probably x[0]...

    f, ax = plt.subplots(8, 8, figsize=(12, 8))
    f.subplots_adjust(hspace=0.1, wspace=0.1)

    cmap = 'copper'
    norm = matplotlib.colors.PowerNorm(gamma=0.5)

    for a in ax.flat:
        a.axis('off')

    for i in range(8):
        for j in range(8):
            if j % 2 == 0:
                field = true_images[i][j//2][...,0]
                ax[i, j].imshow(field, cmap=cmap, norm=norm)
            else:
                field = fake_images[i][j//2][...,0]
                ax[i, j].imshow(field, cmap=cmap, norm=norm)

    plt.savefig('imgs/ms_{0:06d}.png'.format(epoch), dpi=250)
    plt.close()

# ---------------------------------------------------------------------------------------