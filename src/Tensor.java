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