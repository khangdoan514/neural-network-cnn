package src;

public class BCEWithLogitsLoss extends Module {
    public Tensor forward(Tensor logits, Tensor target) {
        Tensor absoluteLogits = logits.relu().add(logits.negate().relu());
        Tensor loss = logits.relu().subtract(logits.multiply(target)).add(absolute_logits.negate().exp().add(1.0).log());

        return loss.mean();
    }

    public Tensor forward(Tensor logits, double[][] target) {
        return forward(logits, new Tensor(target, false));
    }
}