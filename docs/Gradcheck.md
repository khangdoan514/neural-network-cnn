# **Gradcheck**

Compare analytic `.grad` from reverse mode against a central finite difference. If they disagree, the local `gradientFunction` is wrong.

> *`src/Gradcheck.java`*

---

## **Math**

For a scalar $f(x)$:

$$
\partial_i f(x) \approx \frac{f(x+\varepsilon e_i) - f(x-\varepsilon e_i)}{2\varepsilon}
$$

Relative error:

$$
\frac{\lVert g_{\mathrm{analytic}} - g_{\mathrm{numeric}}\rVert}{\lVert g_{\mathrm{analytic}}\rVert + \lVert g_{\mathrm{numeric}}\rVert + 10^{-12}}
$$

---

## **`finiteDifferenceGradient()`**

```java
public static double[] finiteDifferenceGradient(Function<double[], Double> function, double[] data, double epsilon) {
    double[] gradient = new double[data.length];
    double[] probe = Arrays.copyOf(data, data.length);
    for (int index = 0; index < data.length; index++) {
        double original = probe[index];
        probe[index] = original + epsilon;
        double valuePlus = function.apply(probe);
        probe[index] = original - epsilon;
        double valueMinus = function.apply(probe);
        probe[index] = original;
        gradient[index] = (valuePlus - valueMinus) / (2.0 * epsilon);
    }

    return gradient;
}
```

Explanation:

1. Perturb one entry at a time.
2. Restore the original value so later probes stay independent.
3. Central difference is second order in $\varepsilon$.

---

## **`checkTensorGradient()`**

```java
public static Result checkTensorGradient(Function<Tensor, Tensor> makeLoss, Tensor tensor, double epsilon, double absoluteTolerance, double relativeTolerance) {
    tensor.zeroGrad();
    Tensor loss = makeLoss.apply(tensor);
    loss.backward();
    double[] analytic = Arrays.copyOf(tensor.grad, tensor.grad.length);
    Function<double[], Double> function = data -> {
        Tensor probe = new Tensor(data, tensor.shape, false, null, "");
        return makeLoss.apply(probe).data[0];
    };

    double[] numeric = finiteDifferenceGradient(function, Arrays.copyOf(tensor.data, tensor.data.length), epsilon);
    double relativeError = Tensor.norm(subtract(analytic, numeric)) / (Tensor.norm(analytic) + Tensor.norm(numeric) + 1e-12);
    boolean passed = Tensor.allClose(analytic, numeric, absoluteTolerance, relativeTolerance);
    return new Result(passed, relativeError);
}

public static Result checkTensorGradient(Function<Tensor, Tensor> makeLoss, Tensor tensor) {
    return checkTensorGradient(makeLoss, tensor, 1e-5, 1e-4, 1e-3);
}
```

Explanation:

1. Run reverse mode once to fill `tensor.grad`.
2. Re-evaluate the scalar loss on perturbed data for the numeric gradient.
3. Compare with `allClose` and report relative error in `Result`.

Defaults: `epsilon = 1e-5`, `absoluteTolerance = 1e-4`, `relativeTolerance = 1e-3`.

---

## **Check**

```java
Tensor x = new Tensor(new double[] {-1.0, 0.5, 2.0}, true);
Gradcheck.Result result = Gradcheck.checkTensorGradient(tensor -> tensor.relu().power(2).sum(), x);
```

`result.passed` should be true.