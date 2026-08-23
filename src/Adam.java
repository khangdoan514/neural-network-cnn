package src;

public class Adam extends Optimizer {
    public double learningRate;
    public double beta1;
    public double beta2;
    public double epsilon;
    public double weightDecay;
    public int timestep;
    public double[][] firstMoment;
    public double[][] secondMoment;

    public Adam(Iterable<Tensor> parameters, double learningRate, double beta1, double beta2, double epsilon, double weightDecay) {
        super(parameters);
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        this.timestep = 0;
        this.firstMoment = new double[this.parameters.size()][];
        this.secondMoment = new double[this.parameters.size()][];
        for (int index = 0; index < this.parameters.size(); index++) {
            int size = this.parameters.get(index).data.length;
            this.firstMoment[index] = new double[size];
            this.secondMoment[index] = new double[size];
        }
    }

    public Adam(Iterable<Tensor> parameters, double learningRate) {
        this(parameters, learningRate, 0.9, 0.999, 1e-8, 0.0);
    }

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
}