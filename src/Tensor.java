package src;

import java.util.Arrays;
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

    public String representation() {
        String gradInfo = requiresGrad ? ", requiresGrad=" + requiresGrad : "";
        String operationInfo = operation.isEmpty() ? "" : ", operation=" + operation;
        return "Tensor(shape=" + Arrays.toString(shape) + gradInfo + operationInfo + ")\n" + Arrays.toString(data);
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
}