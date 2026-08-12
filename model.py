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

# Step 3 - sample_normal_matrix (not yet solved)
# TODO: implement

# Step 4 - sample_input_features (not yet solved)
# TODO: implement

# Step 5 - assign_class_labels (not yet solved)
# TODO: implement

# Step 6 - one_hot_encode_labels (not yet solved)
# TODO: implement

# Step 7 - init_linear_layer (not yet solved)
# TODO: implement

# Step 8 - init_mlp_params (not yet solved)
# TODO: implement

# Step 9 - linear_forward (not yet solved)
# TODO: implement

# Step 10 - relu_activation (not yet solved)
# TODO: implement

# Step 11 - softmax_probabilities (not yet solved)
# TODO: implement

# Step 12 - mlp_forward (not yet solved)
# TODO: implement

# Step 13 - log_softmax_logits (not yet solved)
# TODO: implement

# Step 14 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 15 - classification_accuracy (not yet solved)
# TODO: implement

# Step 16 - loss_fn_of_params (not yet solved)
# TODO: implement

# Step 17 - compute_param_grads (not yet solved)
# TODO: implement

# Step 18 - sgd_update_params (not yet solved)
# TODO: implement

# Step 19 - training_step (not yet solved)
# TODO: implement

# Step 20 - train_mlp (not yet solved)
# TODO: implement

# Step 21 - predict_classes (not yet solved)
# TODO: implement

