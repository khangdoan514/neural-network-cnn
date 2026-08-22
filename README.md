# **Neural Network CNN**

**CNN** (Convolutional Neural Network) implemented from scratch with **Java** only. The project covers tensors, reverse mode autograd, backpropagation, Conv2d, MaxPool2d, Flatten, linear layers, activations (ReLU, Tanh, Sigmoid, Softmax), losses (MSE, BCE with logits, Cross Entropy), optimizers (SGD with momentum, Adam), minibatch training, finite difference gradient checks, and a LeNet style demo on synthetic MNIST. No ND4J, DJL, PyTorch, TensorFlow, or JAX.

> *The repository is organized as an ordered, step by step curriculum. Each step includes math notes and maps to matching source files.*

---

## **Mathematical Overview**

### **1. Chain rule**

How a scalar loss $L$ pushes gradients backward through $y=g(x)$:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}
$$

### **2. Finite differences**

Numerical check against analytic `.grad`:

$$
\partial_i f(x) \approx \frac{f(x+\varepsilon e_i) - f(x-\varepsilon e_i)}{2\varepsilon}
$$

### **3. Convolution**

2D convolution in NCHW layout $(N, C, H, W)$:

$$
Y[b, c_o, i, j] = \sum_{c_i, u, v} X[b, c_i, i', j'] \cdot W[c_o, c_i, u, v]
$$

with $i' = i \cdot s + u - p$, $j' = j \cdot s + v - p$.

### **4. Max pooling**

$$
Y[b, c, i, j] = \max_{u,v \in \mathcal{W}} X[b, c, i \cdot s + u, j \cdot s + v]
$$

### **5. Affine map**

Forward pass of every `Linear` module:

$$
y = xW + b
$$

with $x \in \mathbb{R}^{N \times d_{\mathrm{in}}}$, $W \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{out}}}$, $b \in \mathbb{R}^{d_{\mathrm{out}}}$.

### **6. Activations**

Elementwise nonlinearities (and softmax over a class axis):

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

### **7. MSE**

Regression objective:

$$
L = \frac{1}{N}\sum_i (\hat{y}_i - y_i)^2
$$

### **8. Cross entropy**

Classification objective from logits:

$$
L = -\log p_c \quad \text{(class index } c\text{)}
$$

### **9. Adam**

Adaptive moments:

$$
\theta \leftarrow \theta - \eta \frac{\hat{m}}{\sqrt{\hat{v}}+\varepsilon}
$$

---

## **Step-by-step Roadmap**

| Step | Topic | Code | Guide |
|------|-------|------|-------|
| 01 | Tensors and ops | `src/Tensor.java` | [docs/Tensors.md](docs/Tensors.md) |
| 02 | Autograd / chain rule | `src/Tensor.java` (`backward`) | [docs/Autograd.md](docs/Autograd.md) |
| 03 | Gradient checking | `src/Gradcheck.java` | [docs/Gradcheck.md](docs/Gradcheck.md) |
| 04 | Conv2d, MaxPool2d, Flatten | `src/Conv2d.java` · `src/MaxPool2d.java` · `src/Flatten.java` | [docs/Layers.md](docs/Layers.md) |
| 05 | Activations | `ReLU`, `Tanh`, `Sigmoid`, `Softmax` | [docs/Activations.md](docs/Activations.md) |
| 06 | Losses | `MSELoss`, `BCEWithLogitsLoss`, `CrossEntropyLoss` | [docs/Losses.md](docs/Losses.md) |
| 07 | Optimizers | `src/SGD.java`, `src/Adam.java` | [docs/Optimizers.md](docs/Optimizers.md) |
| 08 | CNN (`Sequential`) | `src/Sequential.java` | [docs/CNN.md](docs/CNN.md) |
| 09 | Training loop | `src/Training.java` | [docs/Training.md](docs/Training.md) |
| 10 | Experiments | `examples/` | [docs/Experiments.md](docs/Experiments.md) |

---

## **Getting Started**

```bash
git clone https://github.com/khangdoan514/neural-network-cnn
cd neural-network-cnn

mvn test
```

### **Run Demo**

```bash
mvn -q compile
java -cp target/classes examples.TrainMnist
```

### **Minimal Usage**

```java
import src.Conv2d;
import src.CrossEntropyLoss;
import src.MaxPool2d;
import src.ReLU;
import src.Tensor;

Conv2d conv = new Conv2d(1, 4, 3, 1, 1, true, null);
Tensor input = Tensor.randn(new int[] {1, 1, 8, 8}, true, null);
Tensor features = new ReLU().forward(conv.forward(input));
Tensor pooled = new MaxPool2d(2).forward(features);
Tensor logits = new Tensor(new double[][] {{0.1, 0.2, 0.3, 0.0}}, true);
Tensor loss = new CrossEntropyLoss().forward(logits, new int[] {2});
loss.backward();
```

---

## **Project Structure**

```text
neural-network-cnn/
├── src/
│   ├── Tensor.java           # Tensor and autograd ops
│   ├── Gradcheck.java        # Finite difference checks
│   ├── Module.java           # Base module and parameter collection
│   ├── Conv2d.java           # 2D convolution (im2col)
│   ├── MaxPool2d.java        # Max pooling
│   ├── Flatten.java          # NCHW to (N, features)
│   ├── ReLU.java             # Activations
│   ├── Tanh.java
│   ├── Sigmoid.java
│   ├── Softmax.java
│   ├── MSELoss.java          # Losses
│   ├── BCEWithLogitsLoss.java
│   ├── CrossEntropyLoss.java
│   ├── Linear.java           # Dense layer
│   ├── Sequential.java       # Module stack
│   ├── SGD.java              # Optimizers
│   ├── Adam.java
│   ├── Training.java         # Minibatch train loop
│   └── Data.java             # Synthetic MNIST batches
│
├── docs/                     # Step-by-step guides
├── examples/                 # Runnable demos
├── tests/                    # Unit and learning tests
└── README.md
```