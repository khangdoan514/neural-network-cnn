package src;

import java.util.Arrays;

public class MaxPool2d extends Module {
    public int kernelSize;
    public int stride;

    public MaxPool2d(int kernelSize) {
        this(kernelSize, kernelSize);
    }

    public MaxPool2d(int kernelSize, int stride) {
        this.kernelSize = kernelSize;
        this.stride = stride;
    }

    @Override
    public Tensor forward(Tensor input) {
        int batch = input.shape[0];
        int channels = input.shape[1];
        int height = input.shape[2];
        int width = input.shape[3];
        int outHeight = (height - kernelSize) / stride + 1;
        int outWidth = (width - kernelSize) / stride + 1;
        double[] outData = new double[batch * channels * outHeight * outWidth];
        int[] maxIndices = new int[outData.length];
        Arrays.fill(maxIndices, -1);
        for (int batchIndex = 0; batchIndex < batch; batchIndex++) {
            for (int channel = 0; channel < channels; channel++) {
                for (int outRow = 0; outRow < outHeight; outRow++) {
                    for (int outColumn = 0; outColumn < outWidth; outColumn++) {
                        double maximum = Double.NEGATIVE_INFINITY;
                        int bestIndex = -1;
                        for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
                            for (int kernelColumn = 0; kernelColumn < kernelSize; kernelColumn++) {
                                int row = outRow * stride + kernelRow;
                                int column = outColumn * stride + kernelColumn;
                                int inputIndex = Conv2d.nchwIndex(input.shape, batchIndex, channel, row, column);
                                if (input.data[inputIndex] > maximum) {
                                    maximum = input.data[inputIndex];
                                    bestIndex = inputIndex;
                                }
                            }
                        }

                        int outIndex = ((batchIndex * channels + channel) * outHeight + outRow) * outWidth + outColumn;
                        outData[outIndex] = maximum;
                        maxIndices[outIndex] = bestIndex;
                    }
                }
            }
        }

        Tensor out = new Tensor(outData, new int[] {batch, channels, outHeight, outWidth}, input.requiresGrad, new Tensor[] {input}, "maxpool");
        Tensor self = input;
        out.gradientFunction = () -> {
            if (!self.requiresGrad) {
                return;
            }

            for (int index = 0; index < maxIndices.length; index++) {
                int sourceIndex = maxIndices[index];
                if (sourceIndex >= 0) {
                    self.grad[sourceIndex] += out.grad[index];
                }
            }
        };

        return out;
    }
}