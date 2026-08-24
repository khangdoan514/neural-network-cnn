package src;

import java.util.Random;

public class Linear extends Module {
    public Tensor weight;
    public Tensor bias;

    public Linear(int inFeatures, int outFeatures, boolean bias, Random randomGenerator) {
        this.weight = kaiming(inFeatures, outFeatures, randomGenerator);
        this.bias = bias ? Tensor.zeros(new int[] {outFeatures}, true) : null;
    }

    public Linear(int inFeatures, int outFeatures) {
        this(inFeatures, outFeatures, true, null);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor output = input.matmul(weight);
        if (bias != null) {
            output = output.add(bias);
        }

        return output;
    }
}