package src;

public class ReLU extends Module {
    @Override
    public Tensor forward(Tensor input) {
        return input.relu();
    }
}