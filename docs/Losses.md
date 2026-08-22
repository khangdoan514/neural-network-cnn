# **Losses**

Scalar objectives $L(\hat y, y)$ that turn network outputs into something you can call `.backward()` on. Each loss is a `Module` whose `forward` returns a scalar `Tensor`.

> *`src/MSELoss.java` · `src/BCEWithLogitsLoss.java` · `src/CrossEntropyLoss.java`*

---

## **Math**

**MSE** (regression)

$$
L = \frac{1}{N}\sum_i (\hat{y}_i - y_i)^2
$$

**BCE with logits** (binary classification, stable form)

For logits $z$ and targets $y \in \{0,1\}$:

$$
L = \frac{1}{N}\sum_i \Big(\max(z_i,0) - z_i y_i + \log(1 + e^{-|z_i|})\Big)
$$

**Cross entropy** (multi-class, from logits)

For logits $z \in \mathbb{R}^{N \times C}$ and class indices $c_i$:

$$
L = -\frac{1}{N}\sum_i \log \mathrm{softmax}(z_i)_{c_i}
$$

Using log-sum-exp avoids an explicit `Softmax` layer and stays numerically stable.

---

## **`MSELoss`**

Mean squared error: `mean((prediction - target)^2)`.

```java
public class MSELoss extends Module {
    public Tensor forward(Tensor prediction, Tensor target) {
        return prediction.subtract(target).power(2).mean();
    }

    public Tensor forward(Tensor prediction, double[][] target) {
        return forward(prediction, new Tensor(target, false));
    }
}
```

---

## **`BCEWithLogitsLoss`**

Binary cross entropy on raw logits. Uses the stable rewrite above so large positive or negative logits do not overflow.

```java
public class BCEWithLogitsLoss extends Module {
    public Tensor forward(Tensor logits, Tensor target) {
        Tensor absoluteLogits = logits.relu().add(logits.negate().relu());
        Tensor loss = logits.relu().subtract(logits.multiply(target)).add(absolute_logits.negate().exp().add(1.0).log());

        return loss.mean();
    }

    public Tensor forward(Tensor logits, double[][] target) {
        return forward(logits, new Tensor(target, false));
    }
}
```

Explanation:

1. `|z|` is built as `relu(z) + relu(-z)`.
2. Targets should be floats in $\{0,1\}$ with the same shape as `logits`.
3. Prefer this over `Sigmoid` + BCE when training binary classifiers.

---

## **`CrossEntropyLoss`**

Multi-class cross entropy from logits. `logits` has shape $(N, C)$; `targetIndex` has integer class labels of length $N$.

```java
public Tensor forward(Tensor logits, int[] targetIndex) {
    Tensor flatLogits = logits;
    if (logits.ndim() == 3) {
        flatLogits = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2]);
    }

    Tensor maximum = flatLogits.max(1, true);
    Tensor shifted = flatLogits.subtract(maximum);
    Tensor logSumExp = shifted.exp().sum(1, true).log();
    Tensor logProbabilities = shifted.subtract(logSumExp);
    Tensor chosen = logProbabilities.gatherClass(targetIndex);

    return chosen.mean().negate();
}
```

Explanation:

1. Subtract the row max before `exp` / `log` (log-sum-exp trick).
2. `gatherClass` picks the log probability of the true class.
3. Mean over the batch, then negate.

---

## **Check**

```java
Tensor prediction = new Tensor(new double[][] {{0.5}, {-0.2}}, true);
Tensor mse = new MSELoss().forward(prediction, new double[][] {{1.0}, {0.0}});

Tensor logits = new Tensor(new double[][] {{0.2}, {-0.5}, {1.0}}, true);
Gradcheck.Result bceResult = Gradcheck.checkTensorGradient(tensor -> new BCEWithLogitsLoss().forward(tensor, new double[][] {{1.0}, {0.0}, {1.0}}), logits);

Tensor classLogits = new Tensor(new double[][] {{0.1, 0.2, 0.3}, {0.5, -0.1, 0.0}}, true);
Tensor crossEntropy = new CrossEntropyLoss().forward(classLogits, new int[] {2, 0});
crossEntropy.backward();
```

`mse.data[0]`, `bceResult.passed`, and `crossEntropy.data[0]` should all be finite.