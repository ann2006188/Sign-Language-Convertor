# Training Experiments — Results Table

**Dataset:** 3000 samples, 30 classes, 100 samples/class limit, 10 epochs

|   Experiment | Backbone        |   Learning Rate |   Batch Size |   Epochs |   Accuracy |   Precision |   Recall |   F1 Score |
|-------------:|:----------------|----------------:|-------------:|---------:|-----------:|------------:|---------:|-----------:|
|            1 | resnet18        |          0.001  |           32 |       10 |     0.995  |      0.9952 |   0.995  |     0.995  |
|            2 | resnet18        |          0.0001 |           32 |       10 |     0.995  |      0.9952 |   0.995  |     0.995  |
|            3 | efficientnet_b0 |          0.001  |           64 |       10 |     0.9483 |      0.9585 |   0.9483 |     0.9481 |

## Overfitting / Underfitting Analysis

**Exp1 resnet18 lr=0.001 bs=32**
> GOOD FIT — train and val accuracy are close and both high. Model is generalising well.

**Exp2 resnet18 lr=0.0001 bs=32**
> GOOD FIT — train and val accuracy are close and both high. Model is generalising well.

**Exp3 efficientnet_b0 lr=0.001 bs=64**
> GOOD FIT — train and val accuracy are close and both high. Model is generalising well.

