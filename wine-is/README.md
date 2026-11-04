# Wine Classification with a Neural Network (C#, Windows Forms)

A small Windows Forms app that teaches a neural network to recognise 3 types of wine
from 13 chemical measurements. The network is the 3-layer backpropagation library
provided for the course (`NeuralNet`, `INeuron`, `HNeuron`, `ONeuron`). No other machine
learning libraries are used.

**Task:** multi-class classification. Given a wine's 13 measurements, predict which of
3 grape types (cultivars) it comes from.

**Final result:** 34 of 35 correct (97.1%) on wines the network never saw during training.

## Data description

- **Source:** Wine dataset, UCI Machine Learning Repository
  (Aeberhard, S. and Forina, M., 1991). https://doi.org/10.24432/C5PC7J (CC BY 4.0)
- **What it is:** a chemical analysis of wines grown in one region of Italy, made from
  three different grape cultivars.
- **Size:** 178 wines, 13 numeric features, 1 class label (1, 2 or 3).
  The classes have 59, 71 and 48 wines. There are no missing values.
- **Features:** alcohol, malic acid, ash, alcalinity of ash, magnesium, total phenols,
  flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue,
  OD280/OD315 of diluted wines, proline.
- **Why preprocessing is needed:** the features are on very different scales. Proline
  goes up to 1680, while hue stays between 0.48 and 1.71.
- **File:** `wine.csv` (class label in the first column, header row included).

## How to run

1. Open the project in Visual Studio (needs the .NET 10 SDK).
2. Press F5. `wine.csv` is copied next to the program automatically.
3. Use the buttons in this order:

| Button | What it does |
|---|---|
| 1. Train | Trains the network and prints the error every 100 epochs. Shows the results on the training and validation wines. Saves `results/training_log.csv`. |
| 2. Compare settings | Tries 7 hidden-layer sizes, 10 runs each, and compares them on the validation wines. Saves `results/experiments.csv`. |
| 3. Final test | Trains the final network, saves it to `results/wine_model.txt`, and tests it on the 35 test wines. Press it once, after buttons 1 and 2. |
| Clear | Clears the text box. |

## Project files

| File | Purpose |
|---|---|
| `Form1.cs`, `Form1.Designer.cs` | The window: buttons and a text box showing the output |
| `Program.cs` | Starts the app |
| `WineRunner.cs` | What each button does |
| `TextBoxWriter.cs` | Shows the program's printed output in the text box |
| `MlpClassifier.cs` | Uses the professor's `NeuralNet` to train on the whole dataset and to measure error and accuracy |
| `DataLoader.cs` | Reads `wine.csv` and skips broken rows |
| `Preprocessor.cs` | Splits the data, scales the features, one-hot encodes the labels |
| `Metrics.cs` | Accuracy, confusion matrix, precision, recall, F1 |
| `Experiments.cs` | Compares hidden-layer sizes |
| `BackNeural.cs`, `HiddenN.cs`, `InputN.cs`, `OuputN.cs` | The professor's library (not modified) |

## How it works

**Data preparation**
- The 178 wines are split into 107 for training, 36 for validation and 35 for testing.
  Each group has the same mix of wine types.
- Every feature is scaled to the range 0 to 1 using (value - min) / (max - min). The min
  and max come from the training wines only, so nothing about the validation and test
  wines leaks into training.
- Each class label is one-hot encoded, so class 2 becomes (0, 1, 0).

**The network**
- 13 inputs, one hidden layer, 3 outputs. The class with the largest output is the
  prediction.
- It is trained with the library's `learn()` after setting each wine's inputs and correct
  answer, going through the training wines in a new random order every epoch.
- The error shown is the mean squared error between the outputs and the correct answers.
- The library uses fixed learning rates (0.2 for the output layer, 0.15 for the hidden
  layer) and starts every network from random weights, so the learning rate is not tuned.

**Use of the data**
- Training wines: the network learns from these.
- Validation wines: used to compare settings (button 2).
- Test wines: used once at the end (button 3).

## Results

### Training log (button 1)

Network with 2 hidden nodes, 2000 epochs. The full log is in `results/training_log.csv`.

| Epoch | Train error | Validation error | Train accuracy | Validation accuracy |
|---|---|---|---|---|
| 1 | 0.6657 | 0.6675 | 0.3925 | 0.3889 |
| 100 | 0.0232 | 0.0596 | 1.0000 | 1.0000 |
| 500 | 0.0024 | 0.0419 | 1.0000 | 0.9722 |
| 1000 | 0.0011 | 0.0412 | 1.0000 | 0.9722 |
| 2000 | 0.0005 | 0.0412 | 1.0000 | 0.9722 |

The error starts high, because the network is guessing, and falls steadily. Training
accuracy is 100% from epoch 100.

### Choosing the settings (button 2)

Hidden-layer sizes 1, 2, 3, 5, 10, 20 and 50 were each trained 10 times (the library
starts from random weights, so a single run can be lucky or unlucky) and the results were
averaged. The table shows the average validation error and accuracy. All results are in
`results/experiments.csv`.

| Hidden nodes | Epoch 100 | Epoch 500 | Epoch 1000 | Epoch 2000 |
|---|---|---|---|---|
| 1 | 0.3525 / 0.7694 | 0.3028 / 0.9056 | 0.2996 / 0.9194 | 0.2968 / 0.9083 |
| **2** | 0.0637 / 0.9889 | 0.0405 / 0.9861 | 0.0374 / 0.9861 | **0.0353 / 0.9861** |
| 3 | 0.0621 / 0.9833 | 0.0451 / 0.9778 | 0.0423 / 0.9778 | 0.0403 / 0.9750 |
| 5 | 0.0573 / 0.9861 | 0.0444 / 0.9722 | 0.0422 / 0.9722 | 0.0405 / 0.9722 |
| 10 | 0.0577 / 0.9750 | 0.0450 / 0.9722 | 0.0428 / 0.9722 | 0.0410 / 0.9722 |
| 20 | 0.0579 / 0.9750 | 0.0455 / 0.9722 | 0.0430 / 0.9722 | 0.0410 / 0.9722 |
| 50 | 0.0590 / 0.9722 | 0.0470 / 0.9722 | 0.0442 / 0.9722 | 0.0420 / 0.9722 |

(each cell is validation error / validation accuracy)

What this shows:
- **1 hidden node is too small.** It stays at about 91% accuracy.
- **2 hidden nodes did best:** the lowest validation error (0.0353) and the highest
  accuracy (98.6%) at 2000 epochs. Sizes from 3 to 50 are slightly worse, with accuracy
  around 97%, so a bigger network did not help.
- **More epochs helps a little.** The error keeps falling slowly up to 2000 epochs, the
  end of the range tested. The accuracy barely changes after epoch 500.

**Chosen settings: 2 hidden nodes, 2000 epochs.**

### Final test (button 3)

| Wines | Accuracy |
|---|---|
| Training | 1.0000 (107 of 107) |
| Validation | 0.9722 (35 of 36) |
| **Test** | **0.9714 (34 of 35)** |

Test confusion matrix (rows = actual class, columns = predicted class):

| | Predicted 1 | Predicted 2 | Predicted 3 |
|---|---|---|---|
| Actual 1 | 12 | 0 | 0 |
| Actual 2 | 1 | 13 | 0 |
| Actual 3 | 0 | 0 | 9 |

| Class | Precision | Recall | F1 | Wines |
|---|---|---|---|---|
| 1 | 0.9231 | 1.0000 | 0.9600 | 12 |
| 2 | 1.0000 | 0.9286 | 0.9630 | 14 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 9 |
| Average | 0.9744 | 0.9762 | 0.9743 | |

The only mistake is one class 2 wine predicted as class 1. Precision means: of the wines
predicted as this class, how many really are. Recall means: of the wines that really are
this class, how many were found.

## Limitations

- The validation and test sets are small (36 and 35 wines), so one wine is worth about
  3% accuracy. Small differences between settings are within noise.
- The network starts from random weights, so the numbers change a little each time the
  program is run. The saved model in `results/wine_model.txt` is the one from the last
  final test.
- Only the hidden-layer size and the number of epochs were tuned. The learning rate is
  fixed inside the library.
- The final test was run more than once while deciding on the hidden-layer size. I also
  tried 10 hidden nodes on it (35 of 35) but did not use it, because 2 hidden nodes was
  better on the validation wines, which is where the choice should be made. The result
  reported above is for the 2-node setting.
