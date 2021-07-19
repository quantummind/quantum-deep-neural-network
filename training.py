from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import jit
import functools
from jax import random
from jax.scipy.special import erf

import neural_tangents as nt
from neural_tangents import stax

import itertools
import tensorflow_datasets as tfds


"""
Load the MNIST dataset and prepare for training.
"""

def process_data(data_chunk, selection, class_size, flatten_images):
    """
    One-hot encode the labels and normalize the data to the unit sphere.
    Returns tuple of images and labels (+/- 1).
    Arguments:
        data_chunk: either ds['train'] or ds['test']
        selection: tuple of length 2 corresponding to the MNIST digits to classify
        class_size: `int` number of examples to select from each class (MNIST digit)
        flatten_images: `boolean` to return images flattened `shape=(2*class_size, 28**2)`
                        (for fully-connected network) or in 2D form
                        `shape=(2*class_size, 28, 28)` (for convolutional network)
    """
    global key
    image, label = data_chunk['image'], data_chunk['label']
    n_labels = 2

    # pick two labels
    indices = np.where((label == selection[0]) | (label == selection[1]))[0]
    
    key, i_key = random.split(key, 2)
    indices = random.permutation(i_key, indices).reshape(1, -1)
    
    label = (label[tuple(indices)] == selection[0])
    
    # balance if no class size is specified or class size too large
    max_class_size = np.amin(np.unique(label, return_counts=True)[1])
    if (class_size is None) or class_size > max_class_size:
        class_size = max_class_size
        print('class_size', class_size)
    
    # select first class_size examples of each class
    new_indices = []
    for i in range(n_labels):
        class_examples = np.where(label == i)[0]
        new_indices += class_examples[:class_size].tolist()
    new_indices = np.array(new_indices).reshape(1, -1)
    
    label = label[tuple(new_indices)].astype(np.int64)
    label = np.eye(2)[label]
    label = 2*label[:, 0] - 1
    
    image = image[tuple(indices)][tuple(new_indices)]
    image = (image - np.mean(image)) / np.std(image)
    if flatten_images:
        image = image.reshape(image.shape[0], -1)
        norm = np.sqrt(np.sum(image**2, axis=1))
        image /= norm[:, np.newaxis]
    else:
        norm = np.sqrt(np.sum(image**2, axis=(1, 2, 3)))
        image /= norm[:, np.newaxis, np.newaxis, np.newaxis]
    
    return image, label

"""
The following functions are used to compute L_conv = 8*log(n/delta)/mu for the
fully-connected network and prepare the normalized erf function.
"""
def calc_delta(train_xs):
    """
    Returns $1 - |x_i \dot x_j|$ for given dataset `train_xs` of shape (n, d).
    """
    # construct Gramian matrix
    gram = np.tensordot(train_xs, train_xs, axes=[1, 1])
    sep = 1 - np.abs(gram)
    delta = np.amin(sep[~np.eye(sep.shape[0], dtype=bool)])
    return delta

def calc_var(key, s, samples=10**8):
    """
    Computes the variance of activation function s(x).
    """
    key, my_key = random.split(key)
    x = random.normal(my_key, (samples,))
    return np.mean(s(x)**2)

def calc_mu(key, s, samples=10**8):
    """
    Computes the coefficient of nonlinearity for activation function s(x).
    """
    key, my_key = random.split(key)
    x = random.normal(my_key, (samples,))
    return 1 - np.mean(x*s(x))**2

"""
Define both neural network architectures.
"""

def create_fully_connected_network(depth):
    """
    Returns the NTK kernel function of a fully-connected neural network with normalized
    erf functions.
    Arguments:
        depth: number of hidden layers
    """
    layers = []
    for i in range(depth):
        # width can be set to anything since `kernel_fn` is infinite-width
        layers.append(stax.Dense(512, W_std=np.sqrt(1/erf_scale), b_std=0.0))
        layers.append(stax.Erf())

    _, _, kernel_fn = stax.serial(
            *layers,
            stax.Dense(1, W_std=np.sqrt(1/erf_scale), b_std=0.0)
    )
    
    return kernel_fn

def create_convolutional_network(depth):
    """
    Create Myrtle-like architecture 3x3 convolution layers, pool, 3x3 convolution layers,
    pool, 3x3 convolution layers, pool, pool, fully-connected, output. ReLU activation
    functions after convolutions, erf activation function (not normalized) to output.
    Arguments:
        depth: total number of convolutional layers
    """
    # can set width to anything (will be infinite-width when neural-tangents simulates it)
    width = 1
    activation_fn = stax.Relu()
    W_std = np.sqrt(2.0)
    b_std = 0.0
    
    layers = []
    depths = [depth//3, depth//3, depth//3]
    conv = functools.partial(stax.Conv, W_std=W_std, b_std=b_std, padding='SAME')

    layers += [conv(width, (3, 3)), activation_fn] * depths[0]
    layers += [stax.AvgPool((2, 2), strides=(2, 2))]
    layers += [conv(width, (3, 3)), activation_fn] * depths[1]
    layers += [stax.AvgPool((2, 2), strides=(2, 2))]
    layers += [conv(width, (3, 3)), activation_fn] * depths[2]
    layers += [stax.AvgPool((2, 2), strides=(2, 2))] * 2

    layers += [stax.Flatten(), stax.Dense(2, W_std, b_std),
               stax.Erf(), stax.Dense(1, W_std, b_std)]

    _, _, kernel_fn = stax.serial(*layers)
    return kernel_fn

def create_network(depth, network_type):
    """
    Returns neural tangent kernel function.
    Arguments:
        depth: number of hidden or convolutional layers of neural network
        network_type: `'fc'` or `'cnn'` (fully-connected or convolutional neural network)
    """
    if network_type == 'fc':
        return create_fully_connected_network(depth)
    elif network_type == 'cnn':
        return create_convolutional_network(depth)
    else:
        print('INVALID NETWORK TYPE:', network_type)
        return None
    
def get_network_depth(network_type, args=None, fully_connected_depth=0.1,
                      convolutional_depth=96):
    """
    Get depth of neural network for fully-connected (proportional to L_conv) or
    convolutional neural network (constant).
    Arguments:
        network_type: `'fc'` or `'cnn'` (fully-connected or convolutional neural network)
        args: for fully-connected network `(N, delta, mu)`; for convolutional, `None`
        fully_connected_depth: factor to multiply with L_conv for neural network depth
                               convolutional_depth: number of convolutional layers
    """
    if network_type == 'fc':
        N, delta, mu = args
        return round(8*np.log(N/delta)/mu * fully_connected_depth)
    elif network_type == 'cnn':
        return convolutional_depth
    else:
        print('INVALID NETWORK TYPE:', network_type)
        return None
        

"""
Set a normalization factor to prepare the state |k_*> such that none of the components
exceed 1 in magnitude. Here, we show two examples how of how to obtain a normalization
factor, i.e. an upper bound `k_thresh` on the NTK between the test data point (x_*) and
the training set.
    - For the fully-connected network, we pick a \hat\delta estimation of 1 - the inner
      product that is significantly lower than the dataset delta for the training set
      sizes, according to simple extrapolation (see Datasets section of the Supplementary
      Information). We set `k_thresh` to k(1 - \hat\delta).
    - For the convolutional neural network (with fixed depth `L=myrtle_depth`), we set 
      `k_thresh` to the largest value observed from some subset of the training set. While
      this may cause some elements in |k_*> to be clipped, the clipping empirically
      doesn't affect the performance too much.
"""

def compute_k_thresh_from_separability(kernel_fn, data_dim, delta_hat=0.00001):
    """
    Computes k(1 - hat_delta) as an estimate of the maximum NTK. This will be used by the
    fully-connected neural network based on the known dataset separability (Figure S1).
    Arguments:
        kernel_fn: neural tangent kernel function
        delta_hat: minimum separability; must be smaller than expected minimum
                   separability from extrapolation of dataset inner products (estimated
                   from Figure S1)
    """
    # compute k_thresh as k(1 - delta_hat)
    col1 = 1 - delta_hat
    col2 = np.sqrt(1 - col1*col1)
    x0 = [1.0, 0] + [0]*(data_dim-2)
    x1 = [col1, col2] + [0]*(data_dim-2)
    x0 = np.array([x0])
    x1 = np.array([x1])
    iterate_kernel = kernel_fn(x1, x0, 'ntk')
    k_thresh = float(np.abs(iterate_kernel.flatten()[0]))
    return k_thresh

def compute_k_thresh_from_training_set(kernel_fn, selection, k_thresh_n=16):
    """
    Gets largest off-diagonal element of the NTK matrix. For the convolutional neural
    network, this will be used as an estimate of the largest NTK value (since experiments
    are done at a fixed neural network depth).
    Arguments:
        kernel_fn: neural tangent kernel function
        selection: tuple of length 2 corresponding to the MNIST digits to classify
        k_thresh_n: size of subset of training set; we set k_thresh as the largest
                    observed off-diagonal element
    """
    train_image, _ = process_data(ds['train'], selection, k_thresh_n//2, False)
    kernel_fn_batched = nt.utils.batch.batch(kernel_fn, batch_size=1)
    kernel = kernel_fn_batched(train_image, train_image, 'ntk')
    k_thresh = np.amax(np.abs(kernel[np.triu_indices(kernel.shape[0], 1)]))
    return k_thresh

def normalize(m, k_thresh):
    """
    Clips entries of |k_*> to have maximum magnitude k_thresh and sets to range [-1, 1].
    Arguments:
        m: NTK matrix of vectors k_* between test data and training data
        k_thresh: maximum kernel element
    """
    return np.clip(m/k_thresh, -1, 1)


"""
Training the infinite-width neural network.
"""

def get_file_prefix(fp, seed, N, trial):
    """
    NTK output filename
    """
    return fp + '_seed' + str(seed) + '_data' + str(N) + '_trial' + str(trial) + '_'

def train_ntk(network_type, file_prefix, selection):
    """
    Train the infinite-width neural network.
    Arguments:
        network_type: `'fc'` or `'cnn'` (fully-connected or convolutional neural network)
        file_prefix: initila filename
        selection: tuple of length 2 corresponding to the MNIST digits to classify
    """
    
    # pre-compute quantities needed for
    print('Pre-computing quantities...')
    if network_type == 'fc':
        # compute the coefficient of nonlinearity to determine L_conv
        mu = calc_mu(key, lambda x: np.sqrt(1/erf_scale)*erf(x))
    elif network_type == 'cnn':
        # k_thresh is only calculated once
        # select a small training set (n=16) and find the largest off-diagonal element
        cnn_k_thresh = compute_k_thresh_from_training_set(create_network(
                               get_network_depth(network_type), network_type), selection)
    else:
        print('INVALID NETWORK TYPE:', network_type)
        return None
    
    # train the NTK for all dataset sizes
    for i in range(len(data_sizes)):
        N = data_sizes[i]
        print('Training N =', N)
        # bootstrap over many trials
        for t in range(trials[i]):
            # load random dataset from MNIST
            fc = network_type == 'fc'
            train_image, train_label = process_data(ds['train'], selection, N//2, fc)
            test_image, test_label = process_data(ds['test'], selection, test_size//2, fc)
            
            # create neural network
            if network_type == 'fc':
                delta = calc_delta(train_image)
                depth_args = (N, delta, mu)
            else:
                depth_args = None
            depth = get_network_depth(network_type, depth_args)
            kernel_fn = create_network(depth, network_type)
            
            # set the normalization factor
            if network_type == 'fc':
                k_thresh = compute_k_thresh_from_separability(kernel_fn,
                                                              train_image.shape[-1])
            else:
                k_thresh = cnn_k_thresh
            
            # evaluate the NTK on the training set (matrix K) and test set (vector k_*)
            kernel_fn = nt.utils.batch.batch(kernel_fn, batch_size=batch_size)
            kernel_train = kernel_fn(train_image, train_image, 'ntk')
            kernel_test = kernel_fn(test_image, train_image, 'ntk')
            
            # compute the normalized NTK on the test set
            kernel_test_normalized = normalize(kernel_test, k_thresh)
            
            # save everything
            prefix = get_file_prefix(file_prefix, seed, N, t)
            np.save(prefix + 'train_label.npy', train_label)
            np.save(prefix + 'test_label.npy', test_label)
            np.save(prefix + 'kernel_train.npy', kernel_train)
            np.save(prefix + 'kernel_test.npy', kernel_test)
            np.save(prefix + 'kernel_test_normalized.npy', kernel_test_normalized)
            
if __name__ == '__main__':
    """Set up global variables."""
     # initialize random jax random seed
    seed = 0
    key = random.PRNGKey(seed)
    
    # load MNIST dataset
    print('Loading dataset...')
    ds = tfds.as_numpy(
        tfds.load('mnist:3.*.*', batch_size=-1)
    )
    
    # prepare training parameters
    test_size = 256 # number of examples in the test set
    data_sizes = [16, 32, 64, 128, 256, 512] # number of examples in the training set
    
    # to run in reasonable time in the open source implementation, we take only few trials
    trials = [16, 8, 4, 4, 4, 4] # number of subsamples of dataset to bootstrap over
    batch_size = 4 # batch size for training (set according to GPU memory)
    
    print('Normalizing activation function...')
    erf_scale = calc_var(key, erf) # initialize the normalized erf function
    
    
    """Train the neural networks."""
    # train the fully-connected neural network
    print('Training fully-connected network...')
    selection = (8, 9)
    file_prefix = 'kernel_output/fully-connected'
    train_ntk('fc', file_prefix, selection)
    
    # train the convolutional neural network
    print('Training convolutional network...')
    selection = (1, 0)
    file_prefix = 'kernel_output/convolutional'
    train_ntk('cnn', file_prefix, selection)