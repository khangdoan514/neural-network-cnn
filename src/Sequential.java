package src;

import java.util.ArrayList;
import java.util.List;

public class Sequential extends Module {
    public List<Module> layers = new ArrayList<>();

    public Sequential(Module... modules) {
        layers.addAll(List.of(modules));
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor output = input;
        for (Module layer : layers) {
            output = layer.forward(output);
        }

        return output;
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> parameters = new ArrayList<>();
        for (Module layer : layers) {
            parameters.addAll(layer.parameters());
        }

        return parameters;
    }
}