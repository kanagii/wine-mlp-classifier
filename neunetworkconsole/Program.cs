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
            RawDataset data = DataLoader.LoadCsv(path, 0, true); // label is column 0
            ShowSummary(data);

            Console.WriteLine("\nEnd of demo\n");
            Console.ReadLine();
        } // Main

        static void ShowSummary(RawDataset data)
        {
            int numRows = data.Features.Length;
            int numFeatures = data.FeatureNames.Length;
            Console.WriteLine("Rows loaded   : " + numRows);
            Console.WriteLine("Rows skipped  : " + data.NumSkipped);
            Console.WriteLine("Features      : " + numFeatures);
            Console.WriteLine("Classes       : " + data.ClassNames.Length);

            Console.WriteLine("\nItems per class:");
            int[] counts = new int[data.ClassNames.Length];
            for (int i = 0; i < data.Labels.Length; ++i)
                ++counts[data.Labels[i]];
            for (int k = 0; k < counts.Length; ++k)
                Console.WriteLine("  " + data.ClassNames[k] + " (index " + k + "): " + counts[k]);

            Console.WriteLine("\nFeature ranges (raw, unscaled):");
            Console.WriteLine("  " + "name".PadRight(30) + "min".PadLeft(10) +
              "max".PadLeft(10) + "mean".PadLeft(10));
            for (int j = 0; j < numFeatures; ++j)
            {
                double min = double.MaxValue;
                double max = double.MinValue;
                double sum = 0.0;
                for (int i = 0; i < numRows; ++i)
                {
                    double v = data.Features[i][j];
                    if (v < min) min = v;
                    if (v > max) max = v;
                    sum += v;
                }
                Console.WriteLine("  " + data.FeatureNames[j].PadRight(30) +
                  min.ToString("F2").PadLeft(10) + max.ToString("F2").PadLeft(10) +
                  (sum / numRows).ToString("F2").PadLeft(10));
            }
        }
    } // Program
} // ns