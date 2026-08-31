# Build an MLP in JAX from Scratch

Implement a multi-layer perceptron end-to-end in JAX, from PRNG key handling and parameter initialization to forward passes, loss, autodiff, and SGD training. By the end you'll have a fully working classifier trained with jax.grad and pure functional updates.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** make_prng_key
- [x] **2.** split_prng_key
- [x] **3.** sample_normal_matrix
- [x] **4.** sample_input_features
- [x] **5.** assign_class_labels
- [x] **6.** one_hot_encode_labels
- [x] **7.** init_linear_layer
- [x] **8.** init_mlp_params
- [x] **9.** linear_forward
- [x] **10.** relu_activation
- [x] **11.** softmax_probabilities
- [x] **12.** mlp_forward
- [x] **13.** log_softmax_logits
- [x] **14.** cross_entropy_loss
- [x] **15.** classification_accuracy
- [x] **16.** loss_fn_of_params
- [x] **17.** compute_param_grads
- [x] **18.** sgd_update_params
- [x] **19.** training_step
- [x] **20.** train_mlp
- [x] **21.** predict_classes

## Results

```
Input shape: (32, 8)
Labels[:8]: [0, 0, 2, 3, 1, 0, 3, 0]
One-hot[0]: [1.0, 0.0, 0.0, 0.0]
Num layers: 3
Layer shapes: [((), ()), ((), ()), ((), ())]
Initial loss:     1.3831
Initial accuracy: 0.3125
Final loss:       0.1521
Final accuracy:   1.0000
Preds[:8]:  [0, 0, 2, 3, 1, 0, 3, 0]
Labels[:8]: [0, 0, 2, 3, 1, 0, 3, 0]
```
