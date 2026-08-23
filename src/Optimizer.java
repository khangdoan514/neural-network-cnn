package src;

import java.util.ArrayList;
import java.util.List;

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