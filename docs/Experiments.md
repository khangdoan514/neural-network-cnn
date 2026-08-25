# **Experiments**

Image data helpers and a LeNet style demo that prove the CNN stack works end to end. Data helpers live in `src/Data.java`; runnable scripts live under `examples/`.

> *`src/Data.java` · `examples/TrainMnist.java`*

---

## **Datasets**

### **`makeSyntheticMnist()`**

Generates $28 \times 28$ grayscale blobs with class-dependent centers. No download required.

```java
public static ImageBatch makeSyntheticMnist(int sampleCount, int seed) {
    Random randomGenerator = new Random(seed);
    double[] data = new double[sampleCount * 28 * 28];
    int[] labels = new int[sampleCount];
    for (int sample = 0; sample < sampleCount; sample++) {
        int label = sample % 10;
        labels[sample] = label;
        int centerRow = 6 + label;
        int centerColumn = 6 + label * 2;
        for (int row = 0; row < 28; row++) {
            for (int column = 0; column < 28; column++) {
                double distance = Math.hypot(row - centerRow, column - centerColumn);
                double value = Math.max(0.0, 1.0 - distance / 5.0);
                value += 0.03 * randomGenerator.nextGaussian();
                data[sample * 784 + row * 28 + column] = value;
            }
        }
    }

    return new ImageBatch(new Tensor(data, new int[] {sampleCount, 1, 28, 28}, false, null, ""), labels);
}
```

Returns `ImageBatch` with `images` in NCHW layout $(N, 1, 28, 28)$ and integer `labels`.

### **`loadMnist()`**

Reads standard gzip IDX files from a directory:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`

```java
public static ImageBatch loadMnist(Path dataDirectory, int sampleCount) throws IOException
```

Pixels scale to $[0, 1]$. Place files under `data/mnist/` (see [data/mnist/README.md](../data/mnist/README.md)).

---

## **Demo**

| Script | Model | Target |
|--------|-------|--------|
| `examples/TrainMnist.java` | LeNet on synthetic MNIST | accuracy improves |

```bash
mvn -q compile
java -cp target/classes examples.TrainMnist
```

```java
Data.ImageBatch batch = Data.makeSyntheticMnist(512, 0);
Sequential model = new Sequential(
    new Conv2d(1, 6, 5, 1, 2, true, new Random(0)),
    new ReLU(),
    new MaxPool2d(2),
    new Conv2d(6, 16, 5, 1, 0, true, new Random(0)),
    new ReLU(),
    new MaxPool2d(2),
    new Flatten(),
    new Linear(400, 120),
    new ReLU(),
    new Linear(120, 84),
    new ReLU(),
    new Linear(84, 10)
);
Training.TrainingHistory history = Training.trainImages(
    model,
    new CrossEntropyLoss(),
    new Adam(model.parameters(), 0.01),
    batch.images,
    batch.labels,
    12,
    32,
    true,
    2,
    new Random(0)
);
```

---

## **Check**

```java
Data.ImageBatch batch = Data.makeSyntheticMnist(128, 0);
Sequential model = new Sequential(
    new Conv2d(1, 4, 5, 1, 2, true, new Random(0)),
    new ReLU(),
    new MaxPool2d(2),
    new Flatten(),
    new Linear(784, 10)
);
Tensor logits = model.forward(batch.images);
```

`logits.shape` is `{128, 10}` and `batch.labels.length` is `128`.