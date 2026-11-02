using System;

namespace CodingBackProp
{
  /// <summary>Classification metrics: accuracy, confusion matrix, precision, recall, F1.</summary>
  public static class Metrics
  {
    /// <summary>Predicted class index for every row (rows are features then one-hot targets).</summary>
    public static int[] Predict(NeuralNetwork nn, double[][] data, int numFeatures)
    {
      int[] predicted = new int[data.Length];
      double[] xValues = new double[numFeatures];
      for (int i = 0; i < data.Length; ++i)
      {
        Array.Copy(data[i], xValues, numFeatures);
        double[] y = nn.ComputeOutputs(xValues);
        predicted[i] = NeuralNetwork.MaxIndex(y);
      }
      return predicted;
    }

    /// <summary>cm[actual][predicted] = number of rows with that actual and predicted class.</summary>
    public static int[][] ConfusionMatrix(int[] actual, int[] predicted, int numClasses)
    {
      int[][] cm = new int[numClasses][];
      for (int i = 0; i < numClasses; ++i)
        cm[i] = new int[numClasses];
      for (int i = 0; i < actual.Length; ++i)
        ++cm[actual[i]][predicted[i]];
      return cm;
    }

    public static double Accuracy(int[] actual, int[] predicted)
    {
      int correct = 0;
      for (int i = 0; i < actual.Length; ++i)
        if (actual[i] == predicted[i]) ++correct;
      return (correct * 1.0) / actual.Length;
    }

    /// <summary>Prints accuracy, the confusion matrix, and per-class precision / recall / F1.</summary>
    public static void ShowReport(string title, int[] actual, int[] predicted,
      string[] classNames)
    {
      int numClasses = classNames.Length;
      int[][] cm = ConfusionMatrix(actual, predicted, numClasses);

      int correct = 0;
      for (int k = 0; k < numClasses; ++k) correct += cm[k][k];

      Console.WriteLine("\n=== " + title + " ===");
      Console.WriteLine("Accuracy: " + Accuracy(actual, predicted).ToString("F4") +
        "  (" + correct + " of " + actual.Length + " correct)");

      Console.WriteLine("\nConfusion matrix (rows = actual, columns = predicted):");
      string header = "".PadRight(16);
      for (int k = 0; k < numClasses; ++k)
        header += ("pred " + (k + 1)).PadLeft(9);
      Console.WriteLine(header);
      for (int a = 0; a < numClasses; ++a)
      {
        string line = ("actual " + classNames[a].Replace("class ", "")).PadRight(16);
        for (int p = 0; p < numClasses; ++p)
          line += cm[a][p].ToString().PadLeft(9);
        Console.WriteLine(line);
      }

      Console.WriteLine("\nPer-class metrics:");
      Console.WriteLine("  " + "class".PadRight(10) + "precision".PadLeft(11) +
        "recall".PadLeft(9) + "F1".PadLeft(9) + "support".PadLeft(9));
      double sumP = 0.0, sumR = 0.0, sumF = 0.0;
      for (int k = 0; k < numClasses; ++k)
      {
        int tp = cm[k][k];
        int predictedAsK = 0; // column sum = true positives + false positives
        int actuallyK = 0;    // row sum    = true positives + false negatives
        for (int j = 0; j < numClasses; ++j)
        {
          predictedAsK += cm[j][k];
          actuallyK += cm[k][j];
        }
        double precision = (predictedAsK > 0) ? (tp * 1.0) / predictedAsK : 0.0;
        double recall = (actuallyK > 0) ? (tp * 1.0) / actuallyK : 0.0;
        double f1 = (precision + recall > 0.0) ?
          2.0 * precision * recall / (precision + recall) : 0.0;
        sumP += precision; sumR += recall; sumF += f1;
        Console.WriteLine("  " + classNames[k].PadRight(10) +
          precision.ToString("F4").PadLeft(11) + recall.ToString("F4").PadLeft(9) +
          f1.ToString("F4").PadLeft(9) + actuallyK.ToString().PadLeft(9));
      }
      Console.WriteLine("  " + "macro avg".PadRight(10) +
        (sumP / numClasses).ToString("F4").PadLeft(11) +
        (sumR / numClasses).ToString("F4").PadLeft(9) +
        (sumF / numClasses).ToString("F4").PadLeft(9));
    }
  }
}
