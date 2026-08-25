package examples;

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

public class TrainMnist {
    public static void main(String[] args) {
        Data.ImageBatch batch = Data.makeSyntheticMnist(512, 0);
        Sequential model = new Sequential(
                new Conv2d(1, 6, 5, 1, 2, true, new Random(0)),
                new ReLU(),
                new MaxPool2d(2),
                new Conv2d(6, 16, 5, 1, 0, true, new Random(0)),
                new ReLU(),
                new MaxPool2d(2),
                new Flatten(),
                new Linear(400, 120),
                new ReLU(),
                new Linear(120, 84),
                new ReLU(),
                new Linear(84, 10)
        );
        Training.TrainingHistory history = Training.trainImages(
                model,
                new CrossEntropyLoss(),
                new Adam(model.parameters(), 0.01),
                batch.images,
                batch.labels,
                12,
                32,
                true,
                2,
                new Random(0)
        );

        System.out.printf("final accuracy: %.3f%n", history.metric.get(history.metric.size() - 1));
    }
}