package src;

public class Softmax extends Module {
    public int axis;

    public Softmax() {
        this(-1);
    }

    public Softmax(int axis) {
        this.axis = axis;
    }

    @Override
    public Tensor forward(Tensor input) {
        return input.softmax(axis);
    }
}