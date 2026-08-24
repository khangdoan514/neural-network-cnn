package src;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;

public class Training {
    public static class TrainingHistory {
        public List<Double> loss = new ArrayList<>();
        public List<Double> metric = new ArrayList<>();
    }

    public static double accuracyMulticlass(Tensor logits, int[] target) {
        int rows = logits.shape[0];
        int columns = logits.shape[1];
        int count = 0;
        for (int row = 0; row < rows; row++) {
            int best = 0;
            double bestValue = logits.data[row * columns];
            for (int column = 1; column < columns; column++) {
                double value = logits.data[row * columns + column];
                if (value > bestValue) {
                    bestValue = value;
                    best = column;
                }
            }

            if (best == target[row]) {
                count += 1;
            }
        }

        return (double) count / rows;
    }

    public static TrainingHistory trainImages(Module model, CrossEntropyLoss lossFunction, Optimizer optimizer, Tensor inputs, int[] targets, int epochs, Integer batchSize, boolean verbose, int logEvery, Random randomGenerator) {
        Random generator = randomGenerator == null ? new Random(0) : randomGenerator;
        int sampleCount = inputs.shape[0];
        int resolvedBatchSize = batchSize == null ? sampleCount : batchSize;
        TrainingHistory history = new TrainingHistory();
        for (int epoch = 1; epoch <= epochs; epoch++) {
            int[] permutation = permutation(sampleCount, generator);
            List<Double> batchLosses = new ArrayList<>();
            List<Double> batchMetrics = new ArrayList<>();
            for (int start = 0; start < sampleCount; start += resolvedBatchSize) {
                int end = Math.min(start + resolvedBatchSize, sampleCount);
                Tensor batchInputs = gatherBatch(inputs, permutation, start, end);
                int[] batchTargets = gatherLabels(targets, permutation, start, end);
                optimizer.zeroGrad();
                Tensor prediction = model.forward(batchInputs);
                Tensor loss = lossFunction.forward(prediction, batchTargets);
                loss.backward();
                optimizer.step();
                batchLosses.add(loss.data[0]);
                batchMetrics.add(accuracyMulticlass(prediction.detach(), batchTargets));
            }

            double epochLoss = mean(batchLosses);
            double epochMetric = mean(batchMetrics);
            history.loss.add(epochLoss);
            history.metric.add(epochMetric);
            if (verbose && (epoch % logEvery == 0 || epoch == 1 || epoch == epochs)) {
                System.out.printf("epoch %4d | loss %.6f | metric %.4f%n", epoch, epochLoss, epochMetric);
            }
        }

        return history;
    }

    public static TrainingHistory train(Module model, Module lossFunction, Optimizer optimizer, double[][] inputs, double[][] targets, int epochs, Integer batchSize, BiFunction<Tensor, double[][], Double> metricFunction, Random randomGenerator, boolean verbose, int logEvery) {
        Random generator = randomGenerator == null ? new Random(0) : randomGenerator;
        int sampleCount = inputs.length;
        int resolvedBatchSize = batchSize == null ? sampleCount : batchSize;
        TrainingHistory history = new TrainingHistory();
        for (int epoch = 1; epoch <= epochs; epoch++) {
            int[] permutation = permutation(sampleCount, generator);
            double[][] shuffledInputs = gatherRows(inputs, permutation);
            double[][] shuffledTargets = gatherRows(targets, permutation);
            List<Double> batchLosses = new ArrayList<>();
            List<Double> batchMetrics = new ArrayList<>();
            for (int start = 0; start < sampleCount; start += resolvedBatchSize) {
                int end = Math.min(start + resolvedBatchSize, sampleCount);
                double[][] batchInputs = sliceRows(shuffledInputs, start, end);
                double[][] batchTargets = sliceRows(shuffledTargets, start, end);
                optimizer.zeroGrad();
                Tensor prediction = model.forward(new Tensor(batchInputs, false));
                Tensor loss = invokeLoss(lossFunction, prediction, batchTargets);
                loss.backward();
                optimizer.step();
                batchLosses.add(loss.data[0]);
                if (metricFunction != null) {
                    batchMetrics.add(metricFunction.apply(prediction.detach(), batchTargets));
                }
            }

            double epochLoss = mean(batchLosses);
            double epochMetric = batchMetrics.isEmpty() ? Double.NaN : mean(batchMetrics);
            history.loss.add(epochLoss);
            history.metric.add(epochMetric);
            if (verbose && (epoch % logEvery == 0 || epoch == 1 || epoch == epochs)) {
                if (metricFunction == null) {
                    System.out.printf("epoch %4d | loss %.6f%n", epoch, epochLoss);
                }

                else {
                    System.out.printf("epoch %4d | loss %.6f | metric %.4f%n", epoch, epochLoss, epochMetric);
                }
            }
        }

        return history;
    }

    private static Tensor gatherBatch(Tensor input, int[] permutation, int start, int end) {
        int batch = end - start;
        int channels = input.shape[1];
        int height = input.shape[2];
        int width = input.shape[3];
        int sampleSize = channels * height * width;
        double[] data = new double[batch * sampleSize];
        for (int index = 0; index < batch; index++) {
            int source = permutation[start + index];
            System.arraycopy(input.data, source * sampleSize, data, index * sampleSize, sampleSize);
        }

        return new Tensor(data, new int[] {batch, channels, height, width}, false, null, "");
    }

    private static int[] gatherLabels(int[] labels, int[] permutation, int start, int end) {
        int[] batch = new int[end - start];
        for (int index = 0; index < batch.length; index++) {
            batch[index] = labels[permutation[start + index]];
        }

        return batch;
    }

    private static Tensor invokeLoss(Module lossFunction, Tensor prediction, double[][] targets) {
        if (lossFunction instanceof MSELoss mseLoss) {
            return mseLoss.forward(prediction, targets);
        }

        if (lossFunction instanceof BCEWithLogitsLoss bceLoss) {
            return bceLoss.forward(prediction, targets);
        }

        throw new IllegalArgumentException("unsupported loss for dense targets");
    }

    private static int[] permutation(int sampleCount, Random randomGenerator) {
        int[] order = new int[sampleCount];
        for (int index = 0; index < sampleCount; index++) {
            order[index] = index;
        }

        for (int index = sampleCount - 1; index > 0; index--) {
            int swap = randomGenerator.nextInt(index + 1);
            int temporary = order[index];
            order[index] = order[swap];
            order[swap] = temporary;
        }

        return order;
    }

    private static double[][] gatherRows(double[][] values, int[] order) {
        double[][] out = new double[values.length][];
        for (int index = 0; index < order.length; index++) {
            out[index] = values[order[index]];
        }

        return out;
    }

    private static double[][] sliceRows(double[][] values, int start, int end) {
        double[][] out = new double[end - start][];
        System.arraycopy(values, start, out, 0, end - start);

        return out;
    }

    private static double mean(List<Double> values) {
        double total = 0.0;
        for (double value : values) {
            total += value;
        }

        return total / values.size();
    }
}