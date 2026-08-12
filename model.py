"""
Build an MLP in JAX from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_prng_key
import jax
import jax.numpy as jnp

def make_prng_key(seed):
    # Wrap a Python integer seed into a JAX PRNG key
    return jax.random.PRNGKey(seed)

# Step 2 - split_prng_key
def split_prng_key(key, num):
    # Split key into `num` independent subkeys
    return jax.random.split(key, num)

# Step 3 - sample_normal_matrix
def sample_normal_matrix(key, shape):
    # Draw i.i.d. standard-normal samples with the requested shape
    return jax.random.normal(key, shape)

# Step 4 - sample_input_features
def sample_input_features(key, batch_size, num_features):
    """Sample a (batch_size, num_features) standard-normal feature batch."""
    
    return sample_normal_matrix(key, (batch_size, num_features))

# Step 5 - assign_class_labels
def assign_class_labels(inputs, num_classes):
    # Return an int32 label per row using the first num_classes feature columns
    return jnp.argmax(inputs[:, :num_classes], axis=1).astype(jnp.int32)

# Step 6 - one_hot_encode_labels
def one_hot_encode_labels(labels, num_classes):
    # Convert class indices to a float-valued one-hot matrix
    return jax.nn.one_hot(labels, num_classes)

# Step 7 - init_linear_layer
def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    """Return {'W': (in_dim, out_dim), 'b': (out_dim,)} for one dense layer."""
    
    W = scale * sample_normal_matrix(key, (in_dim, out_dim))
    b = jnp.zeros(out_dim)
    
    return {"W": W, "b": b}

# Step 8 - init_mlp_params
def init_mlp_params(key, layer_sizes, scale=0.1):
    # Split the key into one independent subkey per layer
    keys = split_prng_key(key, len(layer_sizes) - 1)

    params = []

    for i, subkey in enumerate(keys):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        params.append(init_linear_layer(subkey, in_dim, out_dim, scale))

    return params

# Step 9 - linear_forward
def linear_forward(x, layer_params):
    # Compute the affine transformation x @ W + b
    return x @ layer_params["W"] + layer_params["b"]

# Step 10 - relu_activation
def relu_activation(x):
    """Apply the ReLU activation elementwise to a JAX array."""
    
    return jnp.maximum(x, 0)

# Step 11 - softmax_probabilities
def softmax_probabilities(logits):
    # Numerically stable softmax along the last axis
    shifted_logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(shifted_logits)
    return exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)

# Step 12 - mlp_forward
def mlp_forward(params, x):
    # Apply linear + ReLU to all hidden layers
    for layer_params in params[:-1]:
        x = linear_forward(x, layer_params)
        x = relu_activation(x)

    # Apply the final linear layer without an activation
    return linear_forward(x, params[-1])

# Step 13 - log_softmax_logits
def log_softmax_logits(logits):
    # Numerically stable log-softmax along the last axis
    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - max_logits
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(shifted_logits), axis=-1, keepdims=True))
    
    return shifted_logits - log_sum_exp

# Step 14 - cross_entropy_loss
def cross_entropy_loss(logits, one_hot_targets):
    # Compute stable log-probabilities
    log_probs = log_softmax_logits(logits)

    # Cross-entropy for each example, then average over the batch
    loss_per_example = -jnp.sum(one_hot_targets * log_probs, axis=-1)
    return jnp.mean(loss_per_example)

# Step 15 - classification_accuracy
def classification_accuracy(logits, labels):
    """Fraction of rows where argmax(logits) equals the integer label."""
    
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)

# Step 16 - loss_fn_of_params
def loss_fn_of_params(params, x, one_hot_targets):
    # Compute logits from the current parameters
    logits = mlp_forward(params, x)

    # Return the scalar cross-entropy loss
    return cross_entropy_loss(logits, one_hot_targets)

# Step 17 - compute_param_grads
def compute_param_grads(params, x, one_hot_targets):
    # Differentiate the loss with respect to the first argument (params)
    grad_fn = jax.grad(loss_fn_of_params)
    return grad_fn(params, x, one_hot_targets)

# Step 18 - sgd_update_params (not yet solved)
# TODO: implement

# Step 19 - training_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - predict_classes (not yet solved)
# TODO: implement

