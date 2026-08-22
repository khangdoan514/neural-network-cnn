package src;

public class MSELoss extends Module {
    public Tensor forward(Tensor prediction, Tensor target) {
        return prediction.subtract(target).power(2).mean();
    }

    public Tensor forward(Tensor prediction, double[][] target) {
        return forward(prediction, new Tensor(target, false));
    }
}