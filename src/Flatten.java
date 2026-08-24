package src;

public class Flatten extends Module {
    @Override
    public Tensor forward(Tensor input) {
        int batch = input.shape[0];
        int features = 1;
        for (int index = 1; index < input.shape.length; index++) {
            features *= input.shape[index];
        }

        return input.reshape(batch, features);
    }
}