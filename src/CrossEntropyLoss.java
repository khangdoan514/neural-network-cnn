package src;

public class CrossEntropyLoss extends Module {
    public Tensor forward(Tensor logits, int[] targetIndex) {
        Tensor flatLogits = logits;
        if (logits.ndim() == 3) {
            flatLogits = logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2]);
        }

        Tensor maximum = flatLogits.max(1, true);
        Tensor shifted = flatLogits.subtract(maximum);
        Tensor logSumExp = shifted.exp().sum(1, true).log();
        Tensor logProbabilities = shifted.subtract(logSumExp);
        Tensor chosen = logProbabilities.gatherClass(targetIndex);

        return chosen.mean().negate();
    }

    public Tensor forward(Tensor logits, int[][] targetIndex) {
        int rows = targetIndex.length;
        int columns = targetIndex[0].length;
        int[] flat = new int[rows * columns];
        for (int row = 0; row < rows; row++) {
            System.arraycopy(targetIndex[row], 0, flat, row * columns, columns);
        }

        return forward(logits, flat);
    }
}