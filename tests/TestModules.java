package tests;

import java.util.List;

import org.junit.jupiter.api.Test;
import src.Adam;
import src.BCEWithLogitsLoss;
import src.Conv2d;
import src.CrossEntropyLoss;
import src.Flatten;
import src.Gradcheck;
import src.Linear;
import src.MaxPool2d;
import src.MSELoss;
import src.ReLU;
import src.SGD;
import src.Sequential;
import src.Softmax;
import src.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestModules {
    @Test
    void testConv2dShape() {
        Conv2d layer = new Conv2d(1, 6, 5, 1, 2, true, null);
        Tensor output = layer.forward(Tensor.ones(new int[] {2, 1, 28, 28}, false));
        assertEquals(2, output.shape[0]);
        assertEquals(6, output.shape[1]);
        assertEquals(28, output.shape[2]);
        assertEquals(28, output.shape[3]);
    }

    @Test
    void testMaxPool2dShape() {
        MaxPool2d layer = new MaxPool2d(2);
        Tensor output = layer.forward(Tensor.ones(new int[] {2, 6, 28, 28}, false));
        assertEquals(14, output.shape[2]);
        assertEquals(14, output.shape[3]);
    }

    @Test
    void testFlattenShape() {
        Flatten layer = new Flatten();
        Tensor output = layer.forward(Tensor.ones(new int[] {2, 16, 5, 5}, false));
        assertEquals(2, output.shape[0]);
        assertEquals(400, output.shape[1]);
    }

    @Test
    void testConv2dGradcheck() {
        Conv2d layer = new Conv2d(1, 2, 3, 1, 0, true, null);
        Tensor inputTensor = Tensor.randn(new int[] {1, 1, 5, 5}, true, null);
        Gradcheck.Result result = Gradcheck.checkTensorGradient(tensor -> layer.forward(tensor).sum(), inputTensor);
        assertTrue(result.passed, Double.toString(result.relativeError));
    }

    @Test
    void testActivationModules() {
        Tensor inputTensor = new Tensor(new double[] {-1.0, 0.5, 2.0}, true);
        Gradcheck.Result result = Gradcheck.checkTensorGradient(tensor -> new ReLU().forward(tensor).power(2).sum(), inputTensor);
        assertTrue(result.passed, Double.toString(result.relativeError));

        Tensor logits = new Tensor(new double[][] {{1.0, 2.0, 3.0}, {0.1, -0.2, 0.3}}, true);
        Tensor probabilities = new Softmax(1).forward(logits);
        double row0 = probabilities.data[0] + probabilities.data[1] + probabilities.data[2];
        double row1 = probabilities.data[3] + probabilities.data[4] + probabilities.data[5];
        assertTrue(Math.abs(row0 - 1.0) < 1e-9);
        assertTrue(Math.abs(row1 - 1.0) < 1e-9);
    }

    @Test
    void testLossGradcheck() {
        Tensor prediction = new Tensor(new double[][] {{0.5}, {-0.2}}, true);
        Gradcheck.Result mseResult = Gradcheck.checkTensorGradient(tensor -> new MSELoss().forward(tensor, new double[][] {{1.0}, {0.0}}), prediction);
        assertTrue(mseResult.passed, Double.toString(mseResult.relativeError));

        Tensor logits = new Tensor(new double[][] {{0.2}, {-0.5}, {1.0}}, true);
        Gradcheck.Result bceResult = Gradcheck.checkTensorGradient(tensor -> new BCEWithLogitsLoss().forward(tensor, new double[][] {{1.0}, {0.0}, {1.0}}), logits);
        assertTrue(bceResult.passed, Double.toString(bceResult.relativeError));

        Tensor classLogits = new Tensor(new double[][] {{0.1, 0.2, 0.3}, {0.5, -0.1, 0.0}}, true);
        Tensor crossEntropy = new CrossEntropyLoss().forward(classLogits, new int[] {2, 0});
        crossEntropy.backward();
        assertTrue(Double.isFinite(crossEntropy.data[0]));
    }

    @Test
    void testAdamSteps() {
        Tensor parameter = new Tensor(new double[] {1.0, 2.0}, true);
        parameter.grad = new double[] {0.5, -0.25};
        double[] before = parameter.data.clone();
        new Adam(List.of(parameter), 0.1).step();
        assertTrue(Math.abs(parameter.data[0] - before[0]) > 1e-12 || Math.abs(parameter.data[1] - before[1]) > 1e-12);
    }

    @Test
    void testSgdWithMomentum() {
        Tensor parameter = new Tensor(new double[] {1.0, 2.0}, true);
        parameter.grad = new double[] {0.1, 0.1};
        SGD optimizer = new SGD(List.of(parameter), 0.01, 0.9, 0.0);
        double[] before = parameter.data.clone();
        optimizer.step();
        assertTrue(Math.abs(parameter.data[0] - before[0]) > 1e-12);
    }

    @Test
    void testLenetForward() {
        Sequential model = new Sequential(
                new Conv2d(1, 6, 5, 1, 2, true, null),
                new ReLU(),
                new MaxPool2d(2),
                new Conv2d(6, 16, 5, 1, 0, true, null),
                new ReLU(),
                new MaxPool2d(2),
                new Flatten(),
                new Linear(400, 10)
        );
        Tensor output = model.forward(Tensor.ones(new int[] {4, 1, 28, 28}, false));
        assertEquals(4, output.shape[0]);
        assertEquals(10, output.shape[1]);
    }
}