# **Optimizers**

Turn $\nabla_\theta L$ into parameter updates. After `loss.backward()`, call `optimizer.step()` to change each parameter in place.

> *`src/Optimizer.java` · `src/SGD.java` · `src/Adam.java`*

---

## **Math**

**SGD**

$$
\theta \leftarrow \theta - \eta g
$$

**SGD with momentum**

$$
v \leftarrow \mu v + g, \qquad \theta \leftarrow \theta - \eta v
$$

**Adam**

$$
m \leftarrow \beta_1 m + (1-\beta_1)g
$$

$$
v \leftarrow \beta_2 v + (1-\beta_2)g^{2}
$$

$$
\hat{m} = \frac{m}{1-\beta_1^{t}}, \quad \hat{v} = \frac{v}{1-\beta_2^{t}}
$$

$$
\theta \leftarrow \theta - \eta \frac{\hat{m}}{\sqrt{\hat{v}}+\varepsilon}
$$

Optional L2 weight decay adds $\lambda \theta$ to $g$ before the update.

---

## **The `Optimizer` Class**

Shared bookkeeping for every optimizer.

```java
public class Optimizer {
    public List<Tensor> parameters;

    public Optimizer(Iterable<Tensor> parameters) {
        this.parameters = new ArrayList<>();
        for (Tensor parameter : parameters) {
            if (parameter.requiresGrad) {
                this.parameters.add(parameter);
            }
        }
    }

    public void zeroGrad() {
        for (Tensor parameter : parameters) {
            parameter.zeroGrad();
        }
    }

    public void step() {
        throw new UnsupportedOperationException();
    }
}
```

Explanation:

1. Only tensors with `requiresGrad` are tracked.
2. `zeroGrad()` clears `.grad` before the next minibatch.
3. Subclasses implement `step()`.

---

## **`SGD`**

Plain SGD, or SGD with momentum when `momentum > 0`.

```java
@Override
public void step() {
    for (int index = 0; index < parameters.size(); index++) {
        Tensor parameter = parameters.get(index);
        if (parameter.grad == null) {
            continue;
        }

        double[] gradient = new double[parameter.grad.length];
        for (int element = 0; element < gradient.length; element++) {
            gradient[element] = parameter.grad[element];
            if (weightDecay != 0.0) {
                gradient[element] += weightDecay * parameter.data[element];
            }
        }

        if (momentum != 0.0) {
            for (int element = 0; element < gradient.length; element++) {
                velocity[index][element] = momentum * velocity[index][element] + gradient[element];
                gradient[element] = velocity[index][element];
            }
        }

        for (int element = 0; element < parameter.data.length; element++) {
            parameter.data[element] -= learningRate * gradient[element];
        }
    }
}
```

---

## **`Adam`**

Adaptive moments with bias correction.

```java
@Override
public void step() {
    timestep += 1;
    for (int index = 0; index < parameters.size(); index++) {
        Tensor parameter = parameters.get(index);
        if (parameter.grad == null) {
            continue;
        }

        double[] gradient = new double[parameter.grad.length];
        for (int element = 0; element < gradient.length; element++) {
            gradient[element] = parameter.grad[element];
            if (weightDecay != 0.0) {
                gradient[element] += weightDecay * parameter.data[element];
            }
        }

        for (int element = 0; element < gradient.length; element++) {
            firstMoment[index][element] = beta1 * firstMoment[index][element] + (1.0 - beta1) * gradient[element];
            secondMoment[index][element] = beta2 * secondMoment[index][element] + (1.0 - beta2) * gradient[element] * gradient[element];
            double firstMomentHat = firstMoment[index][element] / (1.0 - Math.pow(beta1, timestep));
            double secondMomentHat = secondMoment[index][element] / (1.0 - Math.pow(beta2, timestep));
            parameter.data[element] -= learningRate * firstMomentHat / (Math.sqrt(secondMomentHat) + epsilon);
        }
    }
}
```

Explanation:

1. `firstMoment` / `secondMoment` are $m$ and $v$.
2. Bias correction divides by $1-\beta^{t}$ so early steps are not shrunk toward zero.
3. Updates write directly into `parameter.data`; the autograd graph is not involved in the step.

---

## **Check**

```java
Tensor parameter = new Tensor(new double[] {1.0, 2.0}, true);
parameter.grad = new double[] {0.5, -0.25};
double[] before = parameter.data.clone();
new Adam(List.of(parameter), 0.1).step();
```

`parameter.data` moves away from `before`.