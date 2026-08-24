# **CNN**

Stack convolution, pooling, flatten, and linear layers into a classifier. `Sequential` runs modules in order and collects every layer's parameters for the optimizer.

> *`src/Sequential.java` · `src/Linear.java` · `src/Flatten.java`*

---

## **Math**

A convolutional net with nonlinearity $\sigma$ and pooling:

$$
f(x) = W_L\,\sigma(\cdots\mathrm{Pool}(\sigma(W_1 \ast x + b_1))\cdots) + b_L
$$

Local convolution $\ast$ replaces full dense connectivity on early layers. Without $\sigma$, stacked affine maps collapse.

Shapes for a batch of images:

- Input: $x \in \mathbb{R}^{N \times C \times H \times W}$ (NCHW)
- After flatten: $\mathbb{R}^{N \times d}$
- Output logits: $f(x) \in \mathbb{R}^{N \times K}$

---

## **The `Sequential` Class**

Chains modules left to right. Overrides `parameters()` so nested `Conv2d` / `Linear` weights are visible to `SGD` / `Adam`.

```java
public class Sequential extends Module {
    public List<Module> layers = new ArrayList<>();

    public Sequential(Module... modules) {
        layers.addAll(List.of(modules));
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor output = input;
        for (Module layer : layers) {
            output = layer.forward(output);
        }

        return output;
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> parameters = new ArrayList<>();
        for (Module layer : layers) {
            parameters.addAll(layer.parameters());
        }

        return parameters;
    }
}
```

Explanation:

1. `forward` is a left fold: each layer's output is the next layer's input.
2. `parameters()` walks `layers` so nested weights reach the optimizer.
3. Activations and pooling have no learnable tensors; they still participate in the graph.

---

## **`Linear`**

Dense affine map $y = xW + b$ after `Flatten`.

```java
@Override
public Tensor forward(Tensor input) {
    Tensor output = input.matmul(weight);
    if (bias != null) {
        output = output.add(bias);
    }

    return output;
}
```

Weights use Kaiming init from `Module.kaiming`.

---

## **`Flatten`**

Merges $(C, H, W)$ into one feature vector per image.

```java
@Override
public Tensor forward(Tensor input) {
    int batch = input.shape[0];
    int features = 1;
    for (int index = 1; index < input.shape.length; index++) {
        features *= input.shape[index];
    }

    return input.reshape(batch, features);
}
```

---

## **Building a CNN**

```java
Sequential model = new Sequential(
    new Conv2d(1, 6, 5, 1, 2, true, randomGenerator),
    new ReLU(),
    new MaxPool2d(2),
    new Conv2d(6, 16, 5, 1, 0, true, randomGenerator),
    new ReLU(),
    new MaxPool2d(2),
    new Flatten(),
    new Linear(400, 120),
    new ReLU(),
    new Linear(120, 84),
    new ReLU(),
    new Linear(84, 10)
);
```

Typical pattern: `Conv2d` → activation → `MaxPool2d` → … → `Flatten` → `Linear` logits (no `Softmax` when the loss folds it in).

Explanation:

1. Two conv blocks shrink spatial size while growing channels.
2. `Flatten` turns $(16, 5, 5)$ into $400$ features.
3. Three linear layers output ten class logits.

---

## **Check**

```java
Sequential model = new Sequential(
    new Conv2d(1, 6, 5, 1, 2, true, null),
    new ReLU(),
    new MaxPool2d(2),
    new Conv2d(6, 16, 5, 1, 0, true, null),
    new ReLU(),
    new MaxPool2d(2),
    new Flatten(),
    new Linear(400, 10)
);
Tensor logits = model.forward(Tensor.ones(new int[] {4, 1, 28, 28}, false));
```

`logits.shape` is `{4, 10}` and `model.parameters()` is non-empty.