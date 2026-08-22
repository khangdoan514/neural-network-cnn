# **Activations**

Elementwise nonlinearities and softmax over a chosen axis. Each module delegates to the matching `Tensor` op.

> *`src/ReLU.java` · `src/Tanh.java` · `src/Sigmoid.java` · `src/Softmax.java`*

---

## **Math**

$$
\mathrm{ReLU}(x) = \max(x, 0)
$$

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$$
\tanh x = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$

$$
\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

Softmax subtracts a max for stability, then `exp` / `sum`.

---

## **Modules**

```java
public class ReLU extends Module {
    @Override
    public Tensor forward(Tensor input) {
        return input.relu();
    }
}
```

`Tanh` and `Sigmoid` are the same one-liner onto `input.tanh()` and `input.sigmoid()`.

```java
public class Softmax extends Module {
    public int axis;

    public Softmax() {
        this(-1);
    }

    public Softmax(int axis) {
        this.axis = axis;
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.softmax(axis);
    }
}
```

---

## **Check**

```java
Tensor logits = new Tensor(new double[][] {{1.0, 2.0, 3.0}, {0.1, -0.2, 0.3}}, true);
Tensor probabilities = new Softmax(1).forward(logits);
```

Each row of `probabilities.data` sums to $1$.

```java
Tensor inputTensor = new Tensor(new double[] {-1.0, 0.5, 2.0}, true);
Gradcheck.Result result = Gradcheck.checkTensorGradient(tensor -> new ReLU().forward(tensor).power(2).sum(), inputTensor);
```

`result.passed` should be true.