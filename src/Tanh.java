package src;

public class Tanh extends Module {
    @Override
    public Tensor forward(Tensor input) {
        return input.tanh();
    }
}