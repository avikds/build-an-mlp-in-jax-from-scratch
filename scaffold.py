"""
Build an MLP in JAX from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo: train a small MLP in JAX from scratch on synthetic data."""
import numpy as np
import jax
import jax.numpy as jnp

from solution import (
    make_prng_key,
    split_prng_key,
    sample_normal_matrix,
    sample_input_features,
    assign_class_labels,
    one_hot_encode_labels,
    init_linear_layer,
    init_mlp_params,
    linear_forward,
    relu_activation,
    softmax_probabilities,
    mlp_forward,
    log_softmax_logits,
    cross_entropy_loss,
    classification_accuracy,
    loss_fn_of_params,
    compute_param_grads,
    sgd_update_params,
    training_step,
    train_mlp,
    predict_classes,
)


if __name__ == "__main__":
    np.random.seed(0)

    # --- Config ---
    seed = 0
    batch_size = 32
    num_features = 8
    num_classes = 4
    hidden_size = 16
    learning_rate = 0.1
    num_epochs = 200

    # --- PRNG setup ---
    root_key = make_prng_key(seed)
    data_key, init_key = split_prng_key(root_key, 2)

    # --- Synthetic data ---
    x = sample_input_features(data_key, batch_size, num_features)
    labels = assign_class_labels(x, num_classes)
    y_onehot = one_hot_encode_labels(labels, num_classes)

    print("Input shape:", x.shape)
    print("Labels[:8]:", np.asarray(labels[:8]).tolist())
    print("One-hot[0]:", np.asarray(y_onehot[0]).tolist())

    # --- Init MLP ---
    layer_sizes = [num_features, hidden_size, hidden_size, num_classes]
    params = init_mlp_params(init_key, layer_sizes, scale=0.1)
    print("Num layers:", len(params))
    print(
        "Layer shapes:",
        [(np.asarray(W).shape, np.asarray(b).shape) for (W, b) in params],
    )

    # --- Forward pass before training ---
    logits0 = mlp_forward(params, x)
    loss0 = cross_entropy_loss(logits0, y_onehot)
    acc0 = classification_accuracy(logits0, labels)
    print(f"Initial loss:     {float(loss0):.4f}")
    print(f"Initial accuracy: {float(acc0):.4f}")

    # --- Train ---
    trained_params = train_mlp(params, x, y_onehot, learning_rate, num_epochs)

    # --- Evaluate ---
    logits_final = mlp_forward(trained_params, x)
    loss_final = cross_entropy_loss(logits_final, y_onehot)
    acc_final = classification_accuracy(logits_final, labels)
    preds = predict_classes(trained_params, x)

    print(f"Final loss:       {float(loss_final):.4f}")
    print(f"Final accuracy:   {float(acc_final):.4f}")
    print("Preds[:8]: ", np.asarray(preds[:8]).tolist())
    print("Labels[:8]:", np.asarray(labels[:8]).tolist())
