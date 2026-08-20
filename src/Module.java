package src;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Module {
    public boolean training = true;

    public List<Tensor> parameters() {
        List<Tensor> parameters = new ArrayList<>();
        collect(this, parameters);
        return parameters;
    }

    public void zeroGrad() {
        for (Tensor parameter : parameters()) {
            parameter.zeroGrad();
        }
    }

    public void trainMode() {
        training = true;
    }

    public void evalMode() {
        training = false;
    }

    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException();
    }

    public Tensor apply(Tensor input) {
        return forward(input);
    }

    private static void collect(Object object, List<Tensor> parameters) {
        Class<?> type = object.getClass();
        while (type != null && type != Object.class) {
            for (Field field : type.getDeclaredFields()) {
                field.setAccessible(true);
                Object value;
                try {
                    value = field.get(object);
                }

                catch (IllegalAccessException exception) {
                    throw new RuntimeException(exception);
                }

                if (value instanceof Tensor tensor && tensor.requiresGrad) {
                    parameters.add(tensor);
                }

                else if (value instanceof Module module) {
                    collect(module, parameters);
                }

                else if (value instanceof List<?> list) {
                    for (Object item : list) {
                        if (item instanceof Module module) {
                            collect(module, parameters);
                        }

                        else if (item instanceof Tensor tensor && tensor.requiresGrad) {
                            parameters.add(tensor);
                        }
                    }
                }
            }

            type = type.getSuperclass();
        }
    }

    public static Tensor kaiming(int inFeatures, int outFeatures, Random randomGenerator) {
        Random generator = randomGenerator == null ? new Random() : randomGenerator;
        double scale = Math.sqrt(2.0 / inFeatures);
        double[] data = new double[inFeatures * outFeatures];
        for (int index = 0; index < data.length; index++) {
            data[index] = generator.nextGaussian() * scale;
        }

        return new Tensor(data, new int[] {inFeatures, outFeatures}, true, null, "");
    }
}