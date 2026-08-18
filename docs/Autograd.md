# **Autograd**

Reverse mode automatic differentiation: build a graph on the forward pass, then push gradients backward with the chain rule. Each op stores a local `gradientFunction` on its output; `backward()` walks the graph in reverse and runs those rules.

> *`src/Tensor.java`*

---

## **Math**

For a composition $L = f(g(x))$:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial g} \frac{\partial g}{\partial x}
$$

Reverse mode:

1. Forward: evaluate ops and record parents on each output tensor.
2. Seed $\frac{\partial L}{\partial L} = 1$ at the scalar loss.
3. Walk the graph backward; each node applies its local rule and accumulates into parent `.grad` buffers.

Local rules for the elementwise ops in part 2:

| Op | Forward | Backward |
|----|---------|----------|
| Add | $c = a + b$ | $\dot a = \dot c$, $\dot b = \dot c$ |
| Sub | $c = a - b$ | $\dot a = \dot c$, $\dot b = -\dot c$ |
| Mul | $c = a \cdot b$ | $\dot a = \dot c \cdot b$, $\dot b = \dot c \cdot a$ |
| Div | $c = a / b$ | $\dot a = \dot c / b$, $\dot b = -\dot c \cdot a / b^{2}$ |
| Pow | $c = a^{n}$ | $\dot a = \dot c \cdot n a^{n-1}$ |

Elementwise ops broadcast operands to a common output shape before applying the rule. Gradients are accumulated back into each parent with the reverse broadcast.

Dot notation $\dot a$ means $\frac{\partial L}{\partial a}$, the upstream gradient flowing into $a$.

---

## **Graph Fields**

| Field | Role |
|-------|------|
| `parents` | Upstream tensors that produced this node |
| `operation` | Debug name of the creating op |
| `gradientFunction` | Local reverse mode rule (closure) |
| `grad` | Accumulator for $\partial L / \partial(\text{this tensor})$ |

`requiresGrad` is true if this tensor was created with it, or if any parent needs gradients, so the graph stays connected.

---

## **`backward()`**

```java
public void backward(double[] gradient) {
    if (!requiresGrad) {
        throw new RuntimeException("backward called on a tensor that does not require grad");
    }

    double[] seed;
    if (gradient == null) {
        if (data.length != 1) {
            throw new RuntimeException("backward for non-scalar requires an explicit gradient");
        }

        seed = new double[] {1.0};
    }

    else {
        if (gradient.length != data.length) {
            throw new IllegalArgumentException("gradient size does not match tensor size");
        }

        seed = Arrays.copyOf(gradient, gradient.length);
    }

    List<Tensor> topo = new ArrayList<>();
    IdentityHashMap<Tensor, Boolean> visited = new IdentityHashMap<>();
    build(this, visited, topo);
    grad = seed;
    for (int index = topo.size() - 1; index >= 0; index--) {
        topo.get(index).gradientFunction.run();
    }
}
```

Explanation:

1. Default seed is $1$ only for a scalar loss.
2. `build` is a DFS topological sort over `parents`.
3. Reverse walk runs each local `gradientFunction`.

---

## **Check**

```java
Tensor a = new Tensor(new double[] {2.0}, true);
Tensor b = new Tensor(new double[] {3.0}, true);
a.multiply(b).backward();
```

`a.grad` is `[3.0]` and `b.grad` is `[2.0]`.