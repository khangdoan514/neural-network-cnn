package src;

public class Sigmoid extends Module {
    @Override
    public Tensor forward(Tensor input) {
        return input.sigmoid();
    }
}