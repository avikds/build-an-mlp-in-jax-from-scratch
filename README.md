# Build an MLP in JAX from Scratch

Implement a multi-layer perceptron end-to-end in JAX, from PRNG key handling and parameter initialization to forward passes, loss, autodiff, and SGD training. By the end you'll have a fully working classifier trained with jax.grad and pure functional updates.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** make_prng_key
- [ ] **2.** split_prng_key
- [ ] **3.** sample_normal_matrix
- [ ] **4.** sample_input_features
- [ ] **5.** assign_class_labels
- [ ] **6.** one_hot_encode_labels
- [ ] **7.** init_linear_layer
- [ ] **8.** init_mlp_params
- [ ] **9.** linear_forward
- [ ] **10.** relu_activation
- [ ] **11.** softmax_probabilities
- [ ] **12.** mlp_forward
- [ ] **13.** log_softmax_logits
- [ ] **14.** cross_entropy_loss
- [ ] **15.** classification_accuracy
- [ ] **16.** loss_fn_of_params
- [ ] **17.** compute_param_grads
- [ ] **18.** sgd_update_params
- [ ] **19.** training_step
- [ ] **20.** train_mlp
- [ ] **21.** predict_classes

---

Built on Deep-ML.
