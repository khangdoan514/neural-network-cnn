package src;

import java.util.Random;

public class Conv2d extends Module {
    public Tensor weight;
    public Tensor bias;
    public int inChannels;
    public int outChannels;
    public int kernelHeight;
    public int kernelWidth;
    public int stride;
    public int padding;

    public Conv2d(int inChannels, int outChannels, int kernelSize, int stride, int padding, boolean bias, Random randomGenerator) {
        this(inChannels, outChannels, kernelSize, kernelSize, stride, padding, bias, randomGenerator);
    }

    public Conv2d(int inChannels, int outChannels, int kernelHeight, int kernelWidth, int stride, int padding, boolean bias, Random randomGenerator) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.stride = stride;
        this.padding = padding;
        Random generator = randomGenerator == null ? new Random() : randomGenerator;
        double scale = Math.sqrt(2.0 / (inChannels * kernelHeight * kernelWidth));
        double[] weightData = new double[outChannels * inChannels * kernelHeight * kernelWidth];
        for (int index = 0; index < weightData.length; index++) {
            weightData[index] = generator.nextGaussian() * scale;
        }

        this.weight = new Tensor(weightData, new int[] {outChannels, inChannels, kernelHeight, kernelWidth}, true, null, "");
        this.bias = bias ? Tensor.zeros(new int[] {outChannels}, true) : null;
    }

    @Override
    public Tensor forward(Tensor input) {
        int batch = input.shape[0];
        int height = input.shape[2];
        int width = input.shape[3];
        int outHeight = (height + 2 * padding - kernelHeight) / stride + 1;
        int outWidth = (width + 2 * padding - kernelWidth) / stride + 1;
        Tensor columns = im2col(input, kernelHeight, kernelWidth, stride, padding, outHeight, outWidth);
        int kernelElements = inChannels * kernelHeight * kernelWidth;
        Tensor kernelMatrix = weight.reshape(outChannels, kernelElements).transpose();
        Tensor output = columns.matmul(kernelMatrix);
        Tensor reshaped = output.reshape(batch, outHeight, outWidth, outChannels).permute(0, 3, 1, 2);
        if (bias != null) {
            return reshaped.add(bias.reshape(1, outChannels, 1, 1));
        }

        return reshaped;
    }

    static Tensor im2col(Tensor input, int kernelHeight, int kernelWidth, int stride, int padding, int outHeight, int outWidth) {
        int batch = input.shape[0];
        int channels = input.shape[1];
        int height = input.shape[2];
        int width = input.shape[3];
        int kernelElements = channels * kernelHeight * kernelWidth;
        int rows = batch * outHeight * outWidth;
        double[] columnData = new double[rows * kernelElements];
        int[] sourceMap = new int[columnData.length];
        int write = 0;
        for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
            for (int outRow = 0; outRow < outHeight; outRow++) {
                for (int outColumn = 0; outColumn < outWidth; outColumn++) {
                    for (int channel = 0; channel < channels; channel++) {
                        for (int kernelRow = 0; kernelRow < kernelHeight; kernelRow++) {
                            for (int kernelColumn = 0; kernelColumn < kernelWidth; kernelColumn++) {
                                int sourceRow = outRow * stride + kernelRow - padding;
                                int sourceColumn = outColumn * stride + kernelColumn - padding;
                                int sourceIndex = -1;
                                double value = 0.0;
                                if (sourceRow >= 0 && sourceRow < height && sourceColumn >= 0 && sourceColumn < width) {
                                    sourceIndex = nchwIndex(input.shape, batchIndex, channel, sourceRow, sourceColumn);
                                    value = input.data[sourceIndex];
                                }

                                columnData[write] = value;
                                sourceMap[write] = sourceIndex;
                                write += 1;
                            }
                        }
                    }
                }
            }
        }

        Tensor columns = new Tensor(columnData, new int[] {rows, kernelElements}, input.requiresGrad, new Tensor[] {input}, "im2col");
        Tensor self = input;
        columns.gradientFunction = () -> {
            if (!self.requiresGrad) {
                return;
            }

            for (int index = 0; index < sourceMap.length; index++) {
                int sourceIndex = sourceMap[index];
                if (sourceIndex >= 0) {
                    self.grad[sourceIndex] += columns.grad[index];
                }
            }
        };

        return columns;
    }

    static int nchwIndex(int[] shape, int batch, int channel, int row, int column) {
        return ((batch * shape[1] + channel) * shape[2] + row) * shape[3] + column;
    }
}