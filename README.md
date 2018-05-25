# MAGAN: Manifold Aligning Generative Adversarial Network
Code for MAGAN ([https://arxiv.org/abs/1803.00385])

Run the toy example with:
```python train.py```

The correspondence loss is specific to each application. To create your own, provide a different implementation for the method ```correspondence_loss``` when initiating MAGAN. It must match the signature of the template provided.