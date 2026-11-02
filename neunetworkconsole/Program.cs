using System;
using System.IO;

namespace CodingBackProp
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nWine classification with a custom 3-layer MLP");

            // wine.csv is copied next to the .exe by the project file
            string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "wine.csv");

            Console.WriteLine("\nLoading " + path);
            RawDataset raw = DataLoader.LoadCsv(path, 0, true); // label is column 0
            Console.WriteLine("Rows loaded: " + raw.Features.Length +
              "   skipped: " + raw.NumSkipped +
              "   features: " + raw.FeatureNames.Length +
              "   classes: " + raw.ClassNames.Length);

            // 60% train, 20% validation, 20% test; features scaled to [0,1]
            int splitSeed = 1;
            SplitDataset data = Preprocessor.Prepare(raw, 0.60, 0.20, splitSeed, true);
            ShowSplitSummary(data);

            Console.WriteLine("\nEnd of demo\n");
            Console.ReadLine();
        } // Main

        static void ShowSplitSummary(SplitDataset data)
        {
            Console.WriteLine("\nSplit sizes (stratified, so class proportions match):");
            ShowPart("Train     ", data.TrainLabels, data.NumClasses);
            ShowPart("Validation", data.ValidationLabels, data.NumClasses);
            ShowPart("Test      ", data.TestLabels, data.NumClasses);

            // training features should span exactly 0..1 because min/max came from them
            double trainMin = double.MaxValue, trainMax = double.MinValue;
            for (int i = 0; i < data.Train.Length; ++i)
                for (int j = 0; j < data.NumFeatures; ++j)
                {
                    double v = data.Train[i][j];
                    if (v < trainMin) trainMin = v;
                    if (v > trainMax) trainMax = v;
                }
            Console.WriteLine("\nScaled training features: min = " +
              trainMin.ToString("F3") + ", max = " + trainMax.ToString("F3"));

            // validation/test were scaled with the TRAINING min/max, so a few values
            // can fall slightly outside 0..1. That is expected, not a bug.
            Console.WriteLine("Scaled values outside [0,1]: validation = " +
              CountOutOfRange(data.Validation, data.NumFeatures) + ", test = " +
              CountOutOfRange(data.Test, data.NumFeatures));

            Console.WriteLine("\nFirst training row (" + data.NumFeatures +
              " scaled features, then " + data.NumClasses + " one-hot targets):");
            ShowRow(data.Train[0]);
        }

        static void ShowPart(string name, int[] labels, int numClasses)
        {
            int[] counts = new int[numClasses];
            for (int i = 0; i < labels.Length; ++i)
                ++counts[labels[i]];
            string s = "  " + name + ": " + labels.Length.ToString().PadLeft(3) + " rows   per class: ";
            for (int k = 0; k < numClasses; ++k)
                s += counts[k] + (k < numClasses - 1 ? " / " : "");
            Console.WriteLine(s);
        }

        static int CountOutOfRange(double[][] rows, int numFeatures)
        {
            int count = 0;
            for (int i = 0; i < rows.Length; ++i)
                for (int j = 0; j < numFeatures; ++j)
                    if (rows[i][j] < 0.0 || rows[i][j] > 1.0) ++count;
            return count;
        }

        static void ShowRow(double[] row)
        {
            string s = "  ";
            for (int j = 0; j < row.Length; ++j)
                s += row[j].ToString("F2") + " ";
            Console.WriteLine(s);
        }
    } // Program
} // ns