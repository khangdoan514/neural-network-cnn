package tests;

import org.junit.jupiter.api.Test;
import src.Adam;
import src.Conv2d;
import src.CrossEntropyLoss;
import src.Data;
import src.Flatten;
import src.Linear;
import src.MaxPool2d;
import src.ReLU;
import src.Sequential;
import src.Training;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class TestLearning {
    @Test
    void testSyntheticMnistLearns() {
        Data.ImageBatch batch = Data.makeSyntheticMnist(128, 0);
        Sequential model = new Sequential(
                new Conv2d(1, 4, 5, 1, 2, true, new Random(0)),
                new ReLU(),
                new MaxPool2d(2),
                new Flatten(),
                new Linear(784, 10)
        );
        Training.TrainingHistory history = Training.trainImages(
                model,
                new CrossEntropyLoss(),
                new Adam(model.parameters(), 0.05),
                batch.images,
                batch.labels,
                20,
                32,
                false,
                10,
                new Random(0)
        );
        assertTrue(history.metric.get(history.metric.size() - 1) >= 0.60);
        assertTrue(history.loss.get(history.loss.size() - 1) < history.loss.get(0));
    }
}