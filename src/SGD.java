package src;

public class SGD extends Optimizer {
    public double learningRate;
    public double momentum;
    public double weightDecay;
    public double[][] velocity;

    public SGD(Iterable<Tensor> parameters, double learningRate, double momentum, double weightDecay) {
        super(parameters);
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.weightDecay = weightDecay;
        this.velocity = new double[this.parameters.size()][];
        for (int index = 0; index < this.parameters.size(); index++) {
            this.velocity[index] = new double[this.parameters.get(index).data.length];
        }
    }

    public SGD(Iterable<Tensor> parameters, double learningRate) {
        this(parameters, learningRate, 0.0, 0.0);
    }

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
}