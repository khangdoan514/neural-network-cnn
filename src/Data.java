package src;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.zip.GZIPInputStream;

public class Data {
    public static class ImageBatch {
        public Tensor images;
        public int[] labels;

        public ImageBatch(Tensor images, int[] labels) {
            this.images = images;
            this.labels = labels;
        }
    }

    public static ImageBatch makeSyntheticMnist(int sampleCount, int seed) {
        Random randomGenerator = new Random(seed);
        double[] data = new double[sampleCount * 28 * 28];
        int[] labels = new int[sampleCount];
        for (int sample = 0; sample < sampleCount; sample++) {
            int label = sample % 10;
            labels[sample] = label;
            int centerRow = 6 + label;
            int centerColumn = 6 + label * 2;
            for (int row = 0; row < 28; row++) {
                for (int column = 0; column < 28; column++) {
                    double distance = Math.hypot(row - centerRow, column - centerColumn);
                    double value = Math.max(0.0, 1.0 - distance / 5.0);
                    value += 0.03 * randomGenerator.nextGaussian();
                    data[sample * 784 + row * 28 + column] = value;
                }
            }
        }

        return new ImageBatch(new Tensor(data, new int[] {sampleCount, 1, 28, 28}, false, null, ""), labels);
    }

    public static ImageBatch loadMnist(Path dataDirectory, int sampleCount) throws IOException {
        Path imagesPath = dataDirectory.resolve("train-images-idx3-ubyte.gz");
        Path labelsPath = dataDirectory.resolve("train-labels-idx1-ubyte.gz");
        if (!Files.exists(imagesPath) || !Files.exists(labelsPath)) {
            throw new IOException("MNIST gzip files not found in " + dataDirectory);
        }

        byte[] imageBytes = readGzip(imagesPath);
        byte[] labelBytes = readGzip(labelsPath);
        int total = readInt(imageBytes, 4);
        int rows = readInt(imageBytes, 8);
        int columns = readInt(imageBytes, 12);
        int count = Math.min(sampleCount, total);
        double[] data = new double[count * rows * columns];
        int imageOffset = 16;
        for (int sample = 0; sample < count; sample++) {
            for (int row = 0; row < rows; row++) {
                for (int column = 0; column < columns; column++) {
                    int value = imageBytes[imageOffset++] & 0xFF;
                    data[sample * rows * columns + row * columns + column] = value / 255.0;
                }
            }
        }

        int[] labels = new int[count];
        int labelOffset = 8;
        for (int sample = 0; sample < count; sample++) {
            labels[sample] = labelBytes[labelOffset++] & 0xFF;
        }

        return new ImageBatch(new Tensor(data, new int[] {count, 1, rows, columns}, false, null, ""), labels);
    }

    private static byte[] readGzip(Path path) throws IOException {
        try (InputStream inputStream = new GZIPInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            return inputStream.readAllBytes();
        }
    }

    private static int readInt(byte[] bytes, int offset) {
        return ((bytes[offset] & 0xFF) << 24)
                | ((bytes[offset + 1] & 0xFF) << 16)
                | ((bytes[offset + 2] & 0xFF) << 8)
                | (bytes[offset + 3] & 0xFF);
    }
}