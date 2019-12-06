# Handwritten-Digit-Classifier

This project uses MNIST handwritten digit database (http://yann.lecun.com/exdb/mnist/).
The dataset contains over 70000 images. Train/Test split used for training the model was 6:1.

The model uses K-Neigbours model i.e. finds the nearest K neigbours using Euclidean distance and predicts the output based on maximum voting strategy.
We also evaluate the model for varying K values and plot the cost-function vs K graph.

The model was initially implemented in an iterative way but was later vectorized for efficiency, reducing the runtime by 65% (4 hours to 40 minutes).
