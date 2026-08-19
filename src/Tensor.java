package src;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Random;

public final class Tensor {
    public double[] data;
    public int[] shape;
    public boolean requiresGrad;
    public double[] grad;
    public Tensor[] parents;
    public String operation;
    public Runnable gradientFunction;

    public Tensor(double[] data, int[] shape, boolean requiresGrad, Tensor[] children, String operation) {
        this.data = Arrays.copyOf(data, data.length);
        this.shape = Arrays.copyOf(shape, shape.length);
        boolean childRequiresGrad = false;
        if (children != null) {
            for (Tensor child : children) {
                if (child.requiresGrad) {
                    childRequiresGrad = true;
                    break;
                }
            }
        }

        this.requiresGrad = requiresGrad || childRequiresGrad;
        this.grad = this.requiresGrad ? new double[this.data.length] : null;
        this.parents = children == null ? new Tensor[0] : children;
        this.operation = operation == null ? "" : operation;
        this.gradientFunction = () -> {};
    }

    public Tensor(double[] data, int[] shape) {
        this(data, shape, false, null, "");
    }

    public Tensor(double[][] matrix, boolean requiresGrad) {
        this(flatten(matrix), new int[] {matrix.length, matrix[0].length}, requiresGrad, null, "");
    }

    public Tensor(double[][] matrix) {
        this(matrix, false);
    }

    public Tensor(double[] vector, boolean requiresGrad) {
        this(vector, new int[] {vector.length}, requiresGrad, null, "");
    }

    public static Tensor zeros(int[] shape, boolean requiresGrad) {
        return new Tensor(new double[product(shape)], shape, requiresGrad, null, "");
    }

    public static Tensor ones(int[] shape, boolean requiresGrad) {
        double[] data = new double[product(shape)];
        Arrays.fill(data, 1.0);
        return new Tensor(data, shape, requiresGrad, null, "");
    }

    public static Tensor randn(int[] shape, boolean requiresGrad, Random randomGenerator) {
        Random generator = randomGenerator == null ? new Random() : randomGenerator;
        double[] data = new double[product(shape)];
        for (int index = 0; index < data.length; index++) {
            data[index] = generator.nextGaussian();
        }

        return new Tensor(data, shape, requiresGrad, null, "");
    }

    public Tensor detach() {
        return new Tensor(data, shape, false, null, "");
    }

    public int[] shape() {
        return Arrays.copyOf(shape, shape.length);
    }

    public int ndim() {
        return shape.length;
    }

    public int size() {
        return data.length;
    }

    public void zeroGrad() {
        if (grad != null) {
            Arrays.fill(grad, 0.0);
        }
    }

    public void backward() {
        backward(null);
    }

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

    private static void build(Tensor node, IdentityHashMap<Tensor, Boolean> visited, List<Tensor> topo) {
        if (visited.containsKey(node)) {
            return;
        }

        visited.put(node, true);
        for (Tensor parent : node.parents) {
            build(parent, visited, topo);
        }

        topo.add(node);
    }

    public Tensor add(Tensor other) {
        return binary(other, (left, right) -> left + right, (gradient, left, right) -> gradient, (gradient, left, right) -> gradient, "add");
    }

    public Tensor add(double scalar) {
        return add(new Tensor(new double[] {scalar}, new int[] {1}));
    }

    public Tensor subtract(Tensor other) {
        return binary(other, (left, right) -> left - right, (gradient, left, right) -> gradient, (gradient, left, right) -> -gradient, "sub");
    }

    public Tensor multiply(Tensor other) {
        return binary(other, (left, right) -> left * right, (gradient, left, right) -> gradient * right, (gradient, left, right) -> gradient * left, "mul");
    }

    public Tensor multiply(double scalar) {
        return multiply(new Tensor(new double[] {scalar}, new int[] {1}));
    }

    public Tensor divide(Tensor other) {
        return binary(other, (left, right) -> left / right, (gradient, left, right) -> gradient / right, (gradient, left, right) -> -gradient * left / (right * right), "div");
    }

    public Tensor divide(double scalar) {
        return divide(new Tensor(new double[] {scalar}, new int[] {1}));
    }

    public Tensor power(double exponent) {
        double[] outData = new double[data.length];
        for (int index = 0; index < data.length; index++) {
            outData[index] = Math.pow(data[index], exponent);
        }

        Tensor out = new Tensor(outData, shape, requiresGrad, new Tensor[] {this}, "pow");
        Tensor self = this;
        out.gradientFunction = () -> {
            if (self.requiresGrad) {
                for (int index = 0; index < self.data.length; index++) {
                    self.grad[index] += out.grad[index] * exponent * Math.pow(self.data[index], exponent - 1.0);
                }
            }
        };

        return out;
    }

    public Tensor negate() {
        return multiply(-1.0);
    }

    public Tensor T() {
        return transpose();
    }

    public Tensor matmul(Tensor other) {
        if (ndim() == 2 && other.ndim() == 2) {
            return matmul2d(other);
        }

        if (ndim() == 3 && other.ndim() == 2) {
            int batch = shape[0];
            int rows = shape[1];
            int shared = shape[2];
            if (other.shape[0] != shared) {
                throw new IllegalArgumentException("matmul inner dimensions do not match");
            }

            int columns = other.shape[1];
            Tensor flat = reshape(batch * rows, shared).matmul2d(other);
            return flat.reshape(batch, rows, columns);
        }

        if (ndim() == 3 && other.ndim() == 3) {
            return batchedMatmul(other);
        }

        if (ndim() == 4 && other.ndim() == 4) {
            if (shape[0] != other.shape[0] || shape[1] != other.shape[1] || shape[3] != other.shape[2]) {
                throw new IllegalArgumentException("matmul inner dimensions do not match");
            }

            int batch = shape[0] * shape[1];
            Tensor left = reshape(batch, shape[2], shape[3]);
            Tensor right = other.reshape(batch, other.shape[2], other.shape[3]);
            return left.matmul(right).reshape(shape[0], shape[1], shape[2], other.shape[3]);
        }

        throw new IllegalArgumentException("unsupported matmul shapes");
    }

    public Tensor transpose() {
        if (ndim() != 2) {
            throw new IllegalArgumentException("transpose without axes requires a matrix");
        }

        return permute(1, 0);
    }

    public Tensor permute(int... axes) {
        int[] newShape = new int[shape.length];
        for (int index = 0; index < axes.length; index++) {
            newShape[index] = shape[axes[index]];
        }

        double[] outData = new double[data.length];
        int[] inStrides = strides(shape);
        int[] outStrides = strides(newShape);
        for (int flat = 0; flat < data.length; flat++) {
            int[] coords = unravel(flat, shape, inStrides);
            int[] permuted = new int[coords.length];
            for (int index = 0; index < axes.length; index++) {
                permuted[index] = coords[axes[index]];
            }

            outData[ravel(permuted, outStrides)] = data[flat];
        }

        Tensor out = new Tensor(outData, newShape, requiresGrad, new Tensor[] {this}, "transpose");
        Tensor self = this;
        int[] inverse = new int[axes.length];
        for (int index = 0; index < axes.length; index++) {
            inverse[axes[index]] = index;
        }

        out.gradientFunction = () -> {
            if (self.requiresGrad) {
                Tensor back = new Tensor(out.grad, out.shape).permute(inverse);
                for (int index = 0; index < self.grad.length; index++) {
                    self.grad[index] += back.data[index];
                }
            }
        };

        return out;
    }

    public Tensor sum() {
        return sum(null, false);
    }

    public Tensor sum(Integer axis, boolean keepdims) {
        if (axis == null) {
            double total = 0.0;
            for (double value : data) {
                total += value;
            }

            Tensor out = new Tensor(new double[] {total}, new int[] {1}, requiresGrad, new Tensor[] {this}, "sum");
            Tensor self = this;
            out.gradientFunction = () -> {
                if (self.requiresGrad) {
                    for (int index = 0; index < self.grad.length; index++) {
                        self.grad[index] += out.grad[0];
                    }
                }
            };

            return out;
        }

        int resolved = axis < 0 ? ndim() + axis : axis;
        int[] newShape = keepdims ? Arrays.copyOf(shape, shape.length) : removeAxis(shape, resolved);
        if (keepdims) {
            newShape[resolved] = 1;
        }

        double[] outData = new double[product(newShape)];
        int[] inStrides = strides(shape);
        int[] outStrides = strides(newShape.length == 0 ? new int[] {1} : newShape);
        for (int flat = 0; flat < data.length; flat++) {
            int[] coords = unravel(flat, shape, inStrides);
            int[] reduced = keepdims ? Arrays.copyOf(coords, coords.length) : removeAxis(coords, resolved);
            if (keepdims) {
                reduced[resolved] = 0;
            }

            int outIndex = newShape.length == 0 ? 0 : ravel(reduced, outStrides);
            outData[outIndex] += data[flat];
        }

        int[] storedShape = newShape.length == 0 ? new int[] {1} : newShape;
        Tensor out = new Tensor(outData, storedShape, requiresGrad, new Tensor[] {this}, "sum");
        Tensor self = this;
        out.gradientFunction = () -> {
            if (!self.requiresGrad) {
                return;
            }

            for (int flat = 0; flat < self.data.length; flat++) {
                int[] coords = unravel(flat, self.shape, inStrides);
                int[] reduced = keepdims ? Arrays.copyOf(coords, coords.length) : removeAxis(coords, resolved);
                if (keepdims) {
                    reduced[resolved] = 0;
                }

                int outIndex = newShape.length == 0 ? 0 : ravel(reduced, outStrides);
                self.grad[flat] += out.grad[outIndex];
            }
        };

        return out;
    }

    public Tensor mean() {
        return sum().divide((double) data.length);
    }

    public Tensor mean(int axis, boolean keepdims) {
        return sum(axis, keepdims).divide((double) shape[axis < 0 ? ndim() + axis : axis]);
    }

    public Tensor reshape(int... newShape) {
        if (product(newShape) != data.length) {
            throw new IllegalArgumentException("reshape size mismatch");
        }

        Tensor out = new Tensor(data, newShape, requiresGrad, new Tensor[] {this}, "reshape");
        Tensor self = this;
        out.gradientFunction = () -> {
            if (self.requiresGrad) {
                for (int index = 0; index < self.grad.length; index++) {
                    self.grad[index] += out.grad[index];
                }
            }
        };

        return out;
    }

    public Tensor relu() {
        double[] outData = new double[data.length];
        for (int index = 0; index < data.length; index++) {
            outData[index] = Math.max(data[index], 0.0);
        }

        Tensor out = new Tensor(outData, shape, requiresGrad, new Tensor[] {this}, "relu");
        Tensor self = this;
        out.gradientFunction = () -> {
            if (self.requiresGrad) {
                for (int index = 0; index < self.data.length; index++) {
                    self.grad[index] += out.grad[index] * (self.data[index] > 0.0 ? 1.0 : 0.0);
                }
            }
        };

        return out;
    }

    public Tensor tanh() {
        double[] values = new double[data.length];
        for (int index = 0; index < data.length; index++) {
            values[index] = Math.tanh(data[index]);
        }

        Tensor out = new Tensor(values, shape, requiresGrad, new Tensor[] {this}, "tanh");
        Tensor self = this;
        out.gradientFunction = () -> {
            if (self.requiresGrad) {
                for (int index = 0; index < self.data.length; index++) {
                    self.grad[index] += out.grad[index] * (1.0 - values[index] * values[index]);
                }
            }
        };

        return out;
    }

    public Tensor gelu() {
        double coefficient = Math.sqrt(2.0 / Math.PI);
        double[] values = new double[data.length];
        for (int index = 0; index < data.length; index++) {
            double x = data[index];
            values[index] = 0.5 * x * (1.0 + Math.tanh(coefficient * (x + 0.044715 * x * x * x)));
        }

        Tensor out = new Tensor(values, shape, requiresGrad, new Tensor[] {this}, "gelu");
        Tensor self = this;
        out.gradientFunction = () -> {
            if (self.requiresGrad) {
                for (int index = 0; index < self.data.length; index++) {
                    double x = self.data[index];
                    double inner = coefficient * (x + 0.044715 * x * x * x);
                    double tanhInner = Math.tanh(inner);
                    double sech2 = 1.0 - tanhInner * tanhInner;
                    double dInner = coefficient * (1.0 + 3.0 * 0.044715 * x * x);
                    self.grad[index] += out.grad[index] * (0.5 * (1.0 + tanhInner) + 0.5 * x * sech2 * dInner);
                }
            }
        };

        return out;
    }

    public Tensor sigmoid() {
        double[] values = new double[data.length];
        for (int index = 0; index < data.length; index++) {
            values[index] = 1.0 / (1.0 + Math.exp(-data[index]));
        }

        Tensor out = new Tensor(values, shape, requiresGrad, new Tensor[] {this}, "sigmoid");
        Tensor self = this;
        out.gradientFunction = () -> {
            if (self.requiresGrad) {
                for (int index = 0; index < self.data.length; index++) {
                    self.grad[index] += out.grad[index] * values[index] * (1.0 - values[index]);
                }
            }
        };

        return out;
    }

    public Tensor exp() {
        double[] values = new double[data.length];
        for (int index = 0; index < data.length; index++) {
            values[index] = Math.exp(data[index]);
        }

        Tensor out = new Tensor(values, shape, requiresGrad, new Tensor[] {this}, "exp");
        Tensor self = this;
        out.gradientFunction = () -> {
            if (self.requiresGrad) {
                for (int index = 0; index < self.data.length; index++) {
                    self.grad[index] += out.grad[index] * values[index];
                }
            }
        };

        return out;
    }

    public Tensor log() {
        double[] values = new double[data.length];
        for (int index = 0; index < data.length; index++) {
            values[index] = Math.log(data[index]);
        }

        Tensor out = new Tensor(values, shape, requiresGrad, new Tensor[] {this}, "log");
        Tensor self = this;
        out.gradientFunction = () -> {
            if (self.requiresGrad) {
                for (int index = 0; index < self.data.length; index++) {
                    self.grad[index] += out.grad[index] / self.data[index];
                }
            }
        };

        return out;
    }

    public Tensor softmax(int axis) {
        int resolved = axis < 0 ? ndim() + axis : axis;
        Tensor maximum = max(resolved, true);
        Tensor shifted = subtract(maximum);
        Tensor ex = shifted.exp();
        Tensor denom = ex.sum(resolved, true);
        return ex.divide(denom);
    }

    public Tensor max(int axis, boolean keepdims) {
        int resolved = axis < 0 ? ndim() + axis : axis;
        int[] newShape = keepdims ? Arrays.copyOf(shape, shape.length) : removeAxis(shape, resolved);
        if (keepdims) {
            newShape[resolved] = 1;
        }

        double[] outData = new double[product(newShape)];
        Arrays.fill(outData, Double.NEGATIVE_INFINITY);
        int[] inStrides = strides(shape);
        int[] outStrides = strides(newShape);
        for (int flat = 0; flat < data.length; flat++) {
            int[] coords = unravel(flat, shape, inStrides);
            int[] reduced = keepdims ? Arrays.copyOf(coords, coords.length) : removeAxis(coords, resolved);
            if (keepdims) {
                reduced[resolved] = 0;
            }

            int outIndex = ravel(reduced, outStrides);
            outData[outIndex] = Math.max(outData[outIndex], data[flat]);
        }

        Tensor out = new Tensor(outData, newShape, requiresGrad, new Tensor[] {this}, "max");
        Tensor self = this;
        out.gradientFunction = () -> {
            if (!self.requiresGrad) {
                return;
            }

            for (int flat = 0; flat < self.data.length; flat++) {
                int[] coords = unravel(flat, self.shape, inStrides);
                int[] reduced = keepdims ? Arrays.copyOf(coords, coords.length) : removeAxis(coords, resolved);
                if (keepdims) {
                    reduced[resolved] = 0;
                }

                int outIndex = ravel(reduced, outStrides);
                if (self.data[flat] == out.data[outIndex]) {
                    self.grad[flat] += out.grad[outIndex];
                }
            }
        };

        return out;
    }

    public Tensor gatherClass(int[] targetIndex) {
        if (ndim() != 2) {
            throw new IllegalArgumentException("gatherClass expects logits of shape (N, C)");
        }

        int rows = shape[0];
        int columns = shape[1];
        double[] outData = new double[rows];
        for (int row = 0; row < rows; row++) {
            outData[row] = data[row * columns + targetIndex[row]];
        }

        Tensor out = new Tensor(outData, new int[] {rows}, requiresGrad, new Tensor[] {this}, "gather");
        Tensor self = this;
        out.gradientFunction = () -> {
            if (self.requiresGrad) {
                for (int row = 0; row < rows; row++) {
                    self.grad[row * columns + targetIndex[row]] += out.grad[row];
                }
            }
        };

        return out;
    }

    public Tensor embeddingLookup(int[][] tokens) {
        if (ndim() != 2) {
            throw new IllegalArgumentException("embedding table must be (V, C)");
        }

        int batch = tokens.length;
        int sequence = tokens[0].length;
        int channels = shape[1];
        double[] outData = new double[batch * sequence * channels];
        for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
            for (int time = 0; time < sequence; time++) {
                int token = tokens[batchIndex][time];
                int source = token * channels;
                int destination = (batchIndex * sequence + time) * channels;
                System.arraycopy(data, source, outData, destination, channels);
            }
        }

        Tensor out = new Tensor(outData, new int[] {batch, sequence, channels}, requiresGrad, new Tensor[] {this}, "embedding");
        Tensor self = this;
        out.gradientFunction = () -> {
            if (!self.requiresGrad) {
                return;
            }

            for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
                for (int time = 0; time < sequence; time++) {
                    int token = tokens[batchIndex][time];
                    int source = token * channels;
                    int destination = (batchIndex * sequence + time) * channels;
                    for (int channel = 0; channel < channels; channel++) {
                        self.grad[source + channel] += out.grad[destination + channel];
                    }
                }
            }
        };

        return out;
    }

    public String representation() {
        String gradInfo = requiresGrad ? ", requiresGrad=" + requiresGrad : "";
        String operationInfo = operation.isEmpty() ? "" : ", operation=" + operation;
        return "Tensor(shape=" + Arrays.toString(shape) + gradInfo + operationInfo + ")\n" + Arrays.toString(data);
    }

    private Tensor binary(Tensor other, DoubleBinary forward, DoubleTernary backwardForA, DoubleTernary backwardForB, String operationName) {
        int[] outShape = broadcastShape(shape, other.shape);
        double[] outData = new double[product(outShape)];
        int[] aStrides = paddedStrides(shape, outShape);
        int[] bStrides = paddedStrides(other.shape, outShape);
        int[] outStrides = strides(outShape);
        for (int flat = 0; flat < outData.length; flat++) {
            int[] coords = unravel(flat, outShape, outStrides);
            double left = data[broadcastIndex(coords, shape, aStrides, outShape.length)];
            double right = other.data[broadcastIndex(coords, other.shape, bStrides, outShape.length)];
            outData[flat] = forward.apply(left, right);
        }

        Tensor out = new Tensor(outData, outShape, requiresGrad || other.requiresGrad, new Tensor[] {this, other}, operationName);
        Tensor leftTensor = this;
        Tensor rightTensor = other;
        out.gradientFunction = () -> {
            if (leftTensor.requiresGrad) {
                accumulateUnbroadcast(leftTensor, out, other, backwardForA, true);
            }

            if (rightTensor.requiresGrad) {
                accumulateUnbroadcast(rightTensor, out, leftTensor, backwardForB, false);
            }
        };

        return out;
    }

    private static void accumulateUnbroadcast(Tensor target, Tensor out, Tensor other, DoubleTernary rule, boolean targetIsLeft) {
        int[] outStrides = strides(out.shape);
        int[] targetStrides = paddedStrides(target.shape, out.shape);
        int[] otherStrides = paddedStrides(other.shape, out.shape);
        double[] incoming = new double[target.data.length];
        for (int flat = 0; flat < out.data.length; flat++) {
            int[] coords = unravel(flat, out.shape, outStrides);
            int targetIndex = broadcastIndex(coords, target.shape, targetStrides, out.shape.length);
            int otherIndex = broadcastIndex(coords, other.shape, otherStrides, out.shape.length);
            double left = targetIsLeft ? target.data[targetIndex] : other.data[otherIndex];
            double right = targetIsLeft ? other.data[otherIndex] : target.data[targetIndex];
            incoming[targetIndex] += rule.apply(out.grad[flat], left, right);
        }

        for (int index = 0; index < target.grad.length; index++) {
            target.grad[index] += incoming[index];
        }
    }

    private Tensor matmul2d(Tensor other) {
        int rows = shape[0];
        int shared = shape[1];
        int columns = other.shape[1];
        if (other.shape[0] != shared) {
            throw new IllegalArgumentException("matmul inner dimensions do not match");
        }

        double[] outData = new double[rows * columns];
        for (int row = 0; row < rows; row++) {
            for (int column = 0; column < columns; column++) {
                double total = 0.0;
                for (int inner = 0; inner < shared; inner++) {
                    total += data[row * shared + inner] * other.data[inner * columns + column];
                }

                outData[row * columns + column] = total;
            }
        }

        Tensor out = new Tensor(outData, new int[] {rows, columns}, requiresGrad || other.requiresGrad, new Tensor[] {this, other}, "matmul");
        Tensor left = this;
        Tensor right = other;
        out.gradientFunction = () -> {
            if (left.requiresGrad) {
                for (int row = 0; row < rows; row++) {
                    for (int inner = 0; inner < shared; inner++) {
                        double total = 0.0;
                        for (int column = 0; column < columns; column++) {
                            total += out.grad[row * columns + column] * right.data[inner * columns + column];
                        }

                        left.grad[row * shared + inner] += total;
                    }
                }
            }

            if (right.requiresGrad) {
                for (int inner = 0; inner < shared; inner++) {
                    for (int column = 0; column < columns; column++) {
                        double total = 0.0;
                        for (int row = 0; row < rows; row++) {
                            total += left.data[row * shared + inner] * out.grad[row * columns + column];
                        }

                        right.grad[inner * columns + column] += total;
                    }
                }
            }
        };

        return out;
    }

    private Tensor batchedMatmul(Tensor other) {
        int batch = shape[0];
        int rows = shape[1];
        int shared = shape[2];
        int columns = other.shape[2];
        double[] outData = new double[batch * rows * columns];
        for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    double total = 0.0;
                    for (int inner = 0; inner < shared; inner++) {
                        total += data[batchIndex * rows * shared + row * shared + inner] * other.data[batchIndex * shared * columns + inner * columns + column];
                    }

                    outData[batchIndex * rows * columns + row * columns + column] = total;
                }
            }
        }

        Tensor out = new Tensor(outData, new int[] {batch, rows, columns}, requiresGrad || other.requiresGrad, new Tensor[] {this, other}, "matmul");
        Tensor left = this;
        Tensor right = other;
        out.gradientFunction = () -> {
            if (left.requiresGrad) {
                for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
                    for (int row = 0; row < rows; row++) {
                        for (int inner = 0; inner < shared; inner++) {
                            double total = 0.0;
                            for (int column = 0; column < columns; column++) {
                                total += out.grad[batchIndex * rows * columns + row * columns + column] * right.data[batchIndex * shared * columns + inner * columns + column];
                            }

                            left.grad[batchIndex * rows * shared + row * shared + inner] += total;
                        }
                    }
                }
            }

            if (right.requiresGrad) {
                for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
                    for (int inner = 0; inner < shared; inner++) {
                        for (int column = 0; column < columns; column++) {
                            double total = 0.0;
                            for (int row = 0; row < rows; row++) {
                                total += left.data[batchIndex * rows * shared + row * shared + inner] * out.grad[batchIndex * rows * columns + row * columns + column];
                            }

                            right.grad[batchIndex * shared * columns + inner * columns + column] += total;
                        }
                    }
                }
            }
        };

        return out;
    }

    public static int product(int[] shape) {
        int total = 1;
        for (int size : shape) {
            total *= size;
        }

        return total;
    }

    public static double[] flatten(double[][] matrix) {
        double[] flat = new double[matrix.length * matrix[0].length];
        int index = 0;
        for (double[] row : matrix) {
            for (double value : row) {
                flat[index++] = value;
            }
        }

        return flat;
    }

    public static boolean allClose(double[] left, double[] right, double absoluteTolerance, double relativeTolerance) {
        if (left.length != right.length) {
            return false;
        }

        for (int index = 0; index < left.length; index++) {
            double difference = Math.abs(left[index] - right[index]);
            double limit = absoluteTolerance + relativeTolerance * Math.abs(right[index]);
            if (difference > limit) {
                return false;
            }
        }

        return true;
    }

    public static double norm(double[] values) {
        double total = 0.0;
        for (double value : values) {
            total += value * value;
        }

        return Math.sqrt(total);
    }

    private static int[] strides(int[] shape) {
        int[] result = new int[shape.length];
        int accumulator = 1;
        for (int index = shape.length - 1; index >= 0; index--) {
            result[index] = accumulator;
            accumulator *= shape[index];
        }

        return result;
    }

    private static int[] unravel(int flat, int[] shape, int[] strides) {
        int[] coords = new int[shape.length];
        int remaining = flat;
        for (int index = 0; index < shape.length; index++) {
            coords[index] = remaining / strides[index];
            remaining %= strides[index];
        }

        return coords;
    }

    private static int ravel(int[] coords, int[] strides) {
        int flat = 0;
        for (int index = 0; index < coords.length; index++) {
            flat += coords[index] * strides[index];
        }

        return flat;
    }

    private static int[] removeAxis(int[] values, int axis) {
        int[] result = new int[values.length - 1];
        int write = 0;
        for (int index = 0; index < values.length; index++) {
            if (index != axis) {
                result[write++] = values[index];
            }
        }

        return result;
    }

    private static int[] broadcastShape(int[] left, int[] right) {
        int ndim = Math.max(left.length, right.length);
        int[] result = new int[ndim];
        for (int index = 0; index < ndim; index++) {
            int leftDim = index < ndim - left.length ? 1 : left[index - (ndim - left.length)];
            int rightDim = index < ndim - right.length ? 1 : right[index - (ndim - right.length)];
            if (leftDim != rightDim && leftDim != 1 && rightDim != 1) {
                throw new IllegalArgumentException("cannot broadcast shapes");
            }

            result[index] = Math.max(leftDim, rightDim);
        }

        return result;
    }

    private static int[] paddedStrides(int[] shape, int[] outShape) {
        int[] padded = new int[outShape.length];
        int offset = outShape.length - shape.length;
        int[] nativeStrides = strides(shape);
        for (int index = 0; index < outShape.length; index++) {
            if (index < offset) {
                padded[index] = 0;
            }

            else {
                int dim = shape[index - offset];
                padded[index] = dim == 1 ? 0 : nativeStrides[index - offset];
            }
        }

        return padded;
    }

    private static int broadcastIndex(int[] coords, int[] shape, int[] paddedStrides, int outNdim) {
        int offset = outNdim - shape.length;
        int flat = 0;
        for (int index = 0; index < outNdim; index++) {
            int coord = coords[index];
            if (index >= offset) {
                int dim = shape[index - offset];
                if (dim == 1) {
                    coord = 0;
                }
            }

            flat += coord * paddedStrides[index];
        }

        return flat;
    }

    @FunctionalInterface
    private interface DoubleBinary {
        double apply(double left, double right);
    }

    @FunctionalInterface
    private interface DoubleTernary {
        double apply(double gradient, double left, double right);
    }
}