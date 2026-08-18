# **Neural Network CNN**

**CNN** (Convolutional Neural Network) implemented from scratch with **Java** only. The project covers tensors, reverse mode autograd, backpropagation, Conv2d, MaxPool2d, Flatten, linear layers, activations (ReLU, Tanh, Sigmoid, Softmax), losses (MSE, BCE with logits, Cross Entropy), optimizers (SGD with momentum, Adam), minibatch training, finite difference gradient checks, and a LeNet style demo on synthetic MNIST. No ND4J, DJL, PyTorch, TensorFlow, or JAX.

> *The repository is organized as an ordered, step by step curriculum. Each step includes math notes and maps to matching source files.*

---

## **Mathematical Overview**

### **1. Chain rule**

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}
$$

### **2. Finite differences**

$$
\partial_i f(x) \approx \frac{f(x+\varepsilon e_i) - f(x-\varepsilon e_i)}{2\varepsilon}
$$

### **3. Convolution**

$$
Y[b, c_o, i, j] = \sum_{c_i, u, v} X[b, c_i, i', j'] \cdot W[c_o, c_i, u, v]
$$

with $i' = i \cdot s + u - p$, $j' = j \cdot s + v - p$.

### **4. Max pooling**

$$
Y[b, c, i, j] = \max_{u,v \in \mathcal{W}} X[b, c, i \cdot s + u, j \cdot s + v]
$$

### **5. Affine map**

$$
y = xW + b
$$

### **6. Cross entropy**

$$
L = -\log p_c \quad \text{(class index } c\text{)}
$$

### **7. Adam**

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
| 04 | Conv2d, MaxPool2d, Flatten | `src/Conv2d.java` | [docs/Layers.md](docs/Layers.md) |
| 05 | Activations | `src/ReLU.java` | [docs/Activations.md](docs/Activations.md) |
| 06 | Losses | `src/CrossEntropyLoss.java` | [docs/Losses.md](docs/Losses.md) |
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
import src.Adam;
import src.Conv2d;
import src.CrossEntropyLoss;
import src.Data;
import src.Flatten;
import src.Linear;
import src.MaxPool2d;
import src.ReLU;
import src.Sequential;
import src.Training;

Data.ImageBatch batch = Data.makeSyntheticMnist(256, 0);
Sequential model = new Sequential(
    new Conv2d(1, 8, 5, 1, 2, true, new java.util.Random(0)),
    new ReLU(),
    new MaxPool2d(2),
    new Flatten(),
    new Linear(1568, 10)
);
Training.trainImages(
    model,
    new CrossEntropyLoss(),
    new Adam(model.parameters(), 0.01),
    batch.images,
    batch.labels,
    10,
    32,
    true,
    2,
    new java.util.Random(0)
);
```

---

## **Project Structure**

```text
neural-network-cnn/
├── src/
│   ├── Tensor.java
│   ├── Gradcheck.java
│   ├── Conv2d.java
│   ├── MaxPool2d.java
│   ├── Flatten.java
│   ├── Linear.java
│   ├── Sequential.java
│   ├── CrossEntropyLoss.java
│   ├── Adam.java
│   ├── Training.java
│   └── Data.java
├── docs/
├── examples/
├── tests/
└── README.md
```