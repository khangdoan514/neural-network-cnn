package tests;

import org.junit.jupiter.api.Test;
import src.Gradcheck;
import src.Tensor;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestTensor {
    @Test
    void testAddMulBackward() {
        Tensor a = new Tensor(new double[] {2.0, 3.0}, true);
        Tensor b = new Tensor(new double[] {4.0, 5.0}, true);
        Tensor loss = a.multiply(b).add(a).sum();
        loss.backward();
        assertTrue(Tensor.allClose(a.grad, new double[] {5.0, 6.0}, 1e-9, 1e-9));
        assertTrue(Tensor.allClose(b.grad, new double[] {2.0, 3.0}, 1e-9, 1e-9));
    }

    @Test
    void testReluGradcheck() {
        Tensor inputTensor = new Tensor(new double[] {-1.0, 0.5, 2.0}, true);
        Gradcheck.Result result = Gradcheck.checkTensorGradient(
                tensor -> tensor.relu().power(2).sum(),
                inputTensor
        );

        assertTrue(result.passed, Double.toString(result.relativeError));
    }

    @Test
    void testSoftmaxSumsToOne() {
        Tensor inputTensor = new Tensor(new double[][] {{1.0, 2.0, 3.0}, {0.1, -0.2, 0.3}}, true);
        Tensor probabilities = inputTensor.softmax(1);
        double row0 = probabilities.data[0] + probabilities.data[1] + probabilities.data[2];
        double row1 = probabilities.data[3] + probabilities.data[4] + probabilities.data[5];
        assertTrue(Math.abs(row0 - 1.0) < 1e-9);
        assertTrue(Math.abs(row1 - 1.0) < 1e-9);
    }
}