from keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU, Input, Activation, BatchNormalization
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.datasets import mnist
import pandas as pd
import numpy as np

from keras_adversarial import AdversarialModel, ImageGridCallback, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling, AdversarialOptimizerAlternating
from image_utils import dim_ordering_fix, dim_ordering_input, dim_ordering_reshape, dim_ordering_unfix

def gan_targets(n):
    """
    Standard training targets [generator_fake, generator_real, discriminator_fake,
    discriminator_real] = [1, 0, 0, 1]
    :param n: number of samples
    :return: array of targets
    """
    generator_fake = np.ones((n, 1))
    generator_real = np.zeros((n, 1))
    discriminator_fake = np.zeros((n, 1))
    discriminator_real = np.ones((n, 1))
    return [generator_fake, generator_real, discriminator_fake, discriminator_real]

def model_generator():
    nch = 256
    
    return Model(g_input, g_V)

def model_discriminator(input_shape=(1, 28, 28), dropout_rate=0.5):
    d_input = dim_ordering_input(input_shape, name="input_x")
    nch = 512
    
    return Model(d_input, d_V)

def mnist_process(x):
    x = x.astype(np.float32) / 255.0
    return x
def mnist_data():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    return mnist_process(xtrain), mnist_process(xtest)


# z in R^100
latent_dim = 100
# x in R^{28x28}
input_shape = (1, 28, 28)
# generator (z -> x)
generator = model_generator()
# discriminator (x -> y)
discriminator = model_discriminator(input_shape=input_shape)
# gan (x - > yfake, yreal), z generated on GPU
gan = simple_gan(generator, discriminator, normal_latent_sampling((latent_dim,)))
# print summary of models
generator.summary()
discriminator.summary()
gan.summary()

# build adversarial model
model = AdversarialModel(base_model=gan, player_params=[generator.trainable_weights, discriminator.trainable_weights])
model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
player_optimizers=[Adam(1e-4, decay=1e-4), Adam(1e-3, decay=1e-4)],
loss='binary_crossentropy')

def generator_sampler():
    zsamples = np.random.normal(size=(10 * 10, latent_dim))
    gen = dim_ordering_unfix(generator.predict(zsamples))
    return gen.reshape((10, 10, 28, 28))

generator_cb = ImageGridCallback(
"output/gan_convolutional/epoch-{:03d}.png",generator_sampler)
xtrain, xtest = mnist_data()
xtrain = dim_ordering_fix(xtrain.reshape((-1, 1, 28, 28)))
xtest = dim_ordering_fix(xtest.reshape((-1, 1, 28, 28)))
y = gan_targets(xtrain.shape[0])
ytest = gan_targets(xtest.shape[0])
history = model.fit(x=xtrain, y=y,
validation_data=(xtest, ytest), callbacks=[generator_cb], nb_epoch=100,
batch_size=32)
df = pd.DataFrame(history.history)
df.to_csv("output/gan_convolutional/history.csv")
generator.save("output/gan_convolutional/generator.h5")
discriminator.save("output/gan_convolutional/discriminator.h5")