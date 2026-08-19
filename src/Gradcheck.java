package src;

import java.util.Arrays;
import java.util.function.Function;

public final class Gradcheck {
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

    private static double[] subtract(double[] left, double[] right) {
        double[] out = new double[left.length];
        for (int index = 0; index < left.length; index++) {
            out[index] = left[index] - right[index];
        }

        return out;
    }

    public static final class Result {
        public final boolean passed;
        public final double relativeError;

        public Result(boolean passed, double relativeError) {
            this.passed = passed;
            this.relativeError = relativeError;
        }
    }
}