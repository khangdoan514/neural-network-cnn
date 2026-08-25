Place the classic MNIST gzip IDX files here:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`

Or load with:

```java
Data.ImageBatch batch = Data.loadMnist(Path.of("data/mnist"), 1000);
```