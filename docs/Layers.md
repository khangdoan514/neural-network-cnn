# **Layers**

Convolution, pooling, and flatten modules for image batches in NCHW layout $(N, C, H, W)$. A `Module` owns learnable `Tensor` parameters; CNN layers extend it.

> *`src/Module.java` · `src/Conv2d.java` · `src/MaxPool2d.java` · `src/Flatten.java`*

---

## **Math**

2D convolution with stride $s$ and padding $p$:

$$
H_{\mathrm{out}} = \left\lfloor \frac{H + 2p - K}{s} \right\rfloor + 1
$$

Max pool with kernel $K$ and stride $s$:

$$
H_{\mathrm{out}} = \left\lfloor \frac{H - K}{s} \right\rfloor + 1
$$

Flatten merges $(C, H, W)$ into one feature vector per image.

---

## **The `Module` Class**

| Method | Role |
|--------|------|
| `parameters()` | Collect every `Tensor` with `requiresGrad` on this module |
| `zeroGrad()` | Zero `.grad` on all parameters |
| `trainMode()` / `evalMode()` | Toggle training flag |
| `forward()` | Define the forward pass |
| `apply()` | Calls `forward()` |

```java
public List<Tensor> parameters() {
    List<Tensor> parameters = new ArrayList<>();
    collect(this, parameters);
    return parameters;
}
```

Explanation:

1. Reflection walks fields the way Python walks `__dict__`.
2. Nested `Module` values and `List`s of modules are collected recursively.

---

## **`Conv2d`**

Implemented with im2col plus `matmul`:

```java
@Override
public Tensor forward(Tensor input) {
    int batch = input.shape[0];
    int height = input.shape[2];
    int width = input.shape[3];
    int outHeight = (height + 2 * padding - kernelHeight) / stride + 1;
    int outWidth = (width + 2 * padding - kernelWidth) / stride + 1;
    Tensor columns = im2col(input, kernelHeight, kernelWidth, stride, padding, outHeight, outWidth);
    int kernelElements = inChannels * kernelHeight * kernelWidth;
    Tensor kernelMatrix = weight.reshape(outChannels, kernelElements).transpose();
    Tensor output = columns.matmul(kernelMatrix);
    Tensor reshaped = output.reshape(batch, outHeight, outWidth, outChannels).permute(0, 3, 1, 2);
    if (bias != null) {
        return reshaped.add(bias.reshape(1, outChannels, 1, 1));
    }

    return reshaped;
}
```

Explanation:

1. Each output location becomes one row of a patch matrix.
2. Convolution becomes a single matrix multiply.
3. Bias broadcasts over spatial dimensions.

---

## **`MaxPool2d`**

```java
public MaxPool2d(int kernelSize, int stride) {
    this.kernelSize = kernelSize;
    this.stride = stride;
}

@Override
public Tensor forward(Tensor input) {
    int batch = input.shape[0];
    int channels = input.shape[1];
    int height = input.shape[2];
    int width = input.shape[3];
    int outHeight = (height - kernelSize) / stride + 1;
    int outWidth = (width - kernelSize) / stride + 1;
    double[] outData = new double[batch * channels * outHeight * outWidth];
    int[] maxIndices = new int[outData.length];
    Arrays.fill(maxIndices, -1);
    for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
        for (int channel = 0; channel < channels; channel++) {
            for (int outRow = 0; outRow < outHeight; outRow++) {
                for (int outColumn = 0; outColumn < outWidth; outColumn++) {
                    double maximum = Double.NEGATIVE_INFINITY;
                    int bestIndex = -1;
                    for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
                        for (int kernelColumn = 0; kernelColumn < kernelSize; kernelColumn++) {
                            int row = outRow * stride + kernelRow;
                            int column = outColumn * stride + kernelColumn;
                            int inputIndex = Conv2d.nchwIndex(input.shape, batchIndex, channel, row, column);
                            if (input.data[inputIndex] > maximum) {
                                maximum = input.data[inputIndex];
                                bestIndex = inputIndex;
                            }
                        }
                    }

                    int outIndex = ((batchIndex * channels + channel) * outHeight + outRow) * outWidth + outColumn;
                    outData[outIndex] = maximum;
                    maxIndices[outIndex] = bestIndex;
                }
            }
        }
    }

    Tensor out = new Tensor(outData, new int[] {batch, channels, outHeight, outWidth}, input.requiresGrad, new Tensor[] {input}, "maxpool");
    Tensor self = input;
    out.gradientFunction = () -> {
        if (!self.requiresGrad) {
            return;
        }

        for (int index = 0; index < maxIndices.length; index++) {
            int sourceIndex = maxIndices[index];
            if (sourceIndex >= 0) {
                self.grad[sourceIndex] += out.grad[index];
            }
        }
    };

    return out;
}
```

Backward routes the upstream gradient only to the argmax index in each window.

---

## **`Flatten`**

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

## **Check**

```java
Conv2d layer = new Conv2d(1, 6, 5, 1, 2, true, null);
Tensor output = layer.forward(Tensor.ones(new int[] {2, 1, 28, 28}, false));
```

`output.shape` is `{2, 6, 28, 28}`.