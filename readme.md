To build, `cmake -B build -DCMAKE_BUILD_TYPE=Release && cd build && make`.

To train on MNIST, `./build/src/mlp --train data/train-images.idx3-ubyte data/train-labels.idx1-ubyte data/t10k-images.idx3-ubyte data/t10k-labels.idx1-ubyte --label_sizes 784 800 10`. To load coefficients, `./build/src/mlp--load t10k-images.idx3-ubyte data/t10k-labels.idx1-ubyte <coefficients path>`.

Performance on MNIST test set: ~1.5% error rate over 100 epochs, similar to vanilla MLP in https://en.wikipedia.org/wiki/MNIST_database#Classifiers.