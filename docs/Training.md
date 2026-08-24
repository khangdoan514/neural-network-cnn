# **Training**

Minibatch loop that ties the model, loss, and optimizer together: shuffle, forward, backward, step, repeat.

> *`src/Training.java`*

---

## **One Step**

For each minibatch:

1. `optimizer.zeroGrad()`
2. Forward: `prediction = model.forward(batch)`
3. `loss = lossFunction.forward(prediction, targets)`
4. `loss.backward()`
5. `optimizer.step()`

An epoch is one full pass over the shuffled dataset. With `batchSize = null`, the whole set is one batch (full-batch GD).

---

## **Metrics**

```java
public static double accuracyMulticlass(Tensor logits, int[] target) {
    int rows = logits.shape[0];
    int columns = logits.shape[1];
    int count = 0;
    for (int row = 0; row < rows; row++) {
        int best = 0;
        double bestValue = logits.data[row * columns];
        for (int column = 1; column < columns; column++) {
            double value = logits.data[row * columns + column];
            if (value > bestValue) {
                bestValue = value;
                best = column;
            }
        }

        if (best == target[row]) {
            count += 1;
        }
    }

    return (double) count / rows;
}
```

Multiclass accuracy takes $\arg\max$ over the class axis.

---

## **`trainImages()`**

Image batches in NCHW layout with integer class labels.

```java
public static TrainingHistory trainImages(Module model, CrossEntropyLoss lossFunction, Optimizer optimizer, Tensor inputs, int[] targets, int epochs, Integer batchSize, boolean verbose, int logEvery, Random randomGenerator) {
    Random generator = randomGenerator == null ? new Random(0) : randomGenerator;
    int sampleCount = inputs.shape[0];
    int resolvedBatchSize = batchSize == null ? sampleCount : batchSize;
    TrainingHistory history = new TrainingHistory();
    for (int epoch = 1; epoch <= epochs; epoch++) {
        int[] permutation = permutation(sampleCount, generator);
        List<Double> batchLosses = new ArrayList<>();
        List<Double> batchMetrics = new ArrayList<>();
        for (int start = 0; start < sampleCount; start += resolvedBatchSize) {
            int end = Math.min(start + resolvedBatchSize, sampleCount);
            Tensor batchInputs = gatherBatch(inputs, permutation, start, end);
            int[] batchTargets = gatherLabels(targets, permutation, start, end);
            optimizer.zeroGrad();
            Tensor prediction = model.forward(batchInputs);
            Tensor loss = lossFunction.forward(prediction, batchTargets);
            loss.backward();
            optimizer.step();
            batchLosses.add(loss.data[0]);
            batchMetrics.add(accuracyMulticlass(prediction.detach(), batchTargets));
        }

        double epochLoss = mean(batchLosses);
        double epochMetric = mean(batchMetrics);
        history.loss.add(epochLoss);
        history.metric.add(epochMetric);
        if (verbose && (epoch % logEvery == 0 || epoch == 1 || epoch == epochs)) {
            System.out.printf("epoch %4d | loss %.6f | metric %.4f%n", epoch, epochLoss, epochMetric);
        }
    }

    return history;
}
```

Explanation:

1. Shuffle indices once per epoch.
2. `gatherBatch` slices NCHW images without copying the full dataset.
3. `accuracyMulticlass` uses `prediction.detach()` so evaluation does not grow the graph.
4. `TrainingHistory` stores per-epoch mean loss and accuracy.

---

## **`train()`**

Dense tabular loop for `(N, features)` inputs and `double[][]` targets (MSE or BCE).

```java
optimizer.zeroGrad();
Tensor prediction = model.forward(new Tensor(batchInputs, false));
Tensor loss = invokeLoss(lossFunction, prediction, batchTargets);
loss.backward();
optimizer.step();
```

Optional `metricFunction` receives detached predictions and batch targets.

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
Training.TrainingHistory history = Training.trainImages(
    model,
    new CrossEntropyLoss(),
    new Adam(model.parameters(), 0.05),
    batch.images,
    batch.labels,
    20,
    32,
    false,
    10,
    new Random(0)
);
```

Final metric should rise and final loss should fall below the first epoch.