using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace CodingBackProp
{
    // the data as read from the csv before scaling or splitting
    public class RawDataset
    {
        public double[][] Features = new double[0][];
        public int[] Labels = new int[0];
        public string[] FeatureNames = new string[0];
        public string[] ClassNames = new string[0];
        public int NumSkipped;
    }

    // reads the wine data from a csv file
    public static class DataLoader
    {
        // reads the csv: one column is the class label and the others are numbers
        // rows with missing or wrong values are skipped and reported
        public static RawDataset LoadCsv(string path, int labelColumn, bool hasHeader)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException("Data file not found: " + path);

            string[] lines = File.ReadAllLines(path);
            if (lines.Length == 0)
                throw new Exception("Data file is empty: " + path);

            string[] firstParts = lines[0].Split(',');
            int numCols = firstParts.Length;
            int numFeatures = numCols - 1;

            string[] featureNames = new string[numFeatures];
            int f = 0;
            for (int j = 0; j < numCols; ++j)
            {
                if (j == labelColumn) continue;
                featureNames[f] = hasHeader ? firstParts[j].Trim() : "feature" + f;
                ++f;
            }

            List<double[]> rows = new List<double[]>();
            List<int> rawLabels = new List<int>();
            int skipped = 0;

            int start = hasHeader ? 1 : 0;
            for (int i = start; i < lines.Length; ++i)
            {
                if (lines[i].Trim().Length == 0) continue;

                string[] parts = lines[i].Split(',');
                if (parts.Length != numCols)
                {
                    Console.WriteLine("Skipping line " + (i + 1) + ": expected " +
                      numCols + " columns but found " + parts.Length);
                    ++skipped;
                    continue;
                }

                double[] row = new double[numFeatures];
                int rawLabel = 0;
                int c = 0;
                bool ok = true;
                for (int j = 0; j < numCols && ok; ++j)
                {
                    string cell = parts[j].Trim();
                    if (j == labelColumn)
                    {
                        ok = int.TryParse(cell, NumberStyles.Integer,
                          CultureInfo.InvariantCulture, out rawLabel);
                    }
                    else
                    {
                        double v;
                        ok = double.TryParse(cell, NumberStyles.Float,
                          CultureInfo.InvariantCulture, out v)
                          && !double.IsNaN(v) && !double.IsInfinity(v);
                        if (ok) row[c++] = v;
                    }
                }

                if (!ok)
                {
                    Console.WriteLine("Skipping line " + (i + 1) +
                      ": missing or non-numeric value");
                    ++skipped;
                    continue;
                }

                rows.Add(row);
                rawLabels.Add(rawLabel);
            }

            // change the labels (1, 2, 3) to class indexes (0, 1, 2)
            SortedSet<int> distinct = new SortedSet<int>(rawLabels);
            List<int> sortedLabels = new List<int>(distinct);
            string[] classNames = new string[sortedLabels.Count];
            for (int k = 0; k < classNames.Length; ++k)
                classNames[k] = "class " + sortedLabels[k];

            int[] labels = new int[rawLabels.Count];
            for (int i = 0; i < labels.Length; ++i)
                labels[i] = sortedLabels.IndexOf(rawLabels[i]);

            RawDataset result = new RawDataset();
            result.Features = rows.ToArray();
            result.Labels = labels;
            result.FeatureNames = featureNames;
            result.ClassNames = classNames;
            result.NumSkipped = skipped;
            return result;
        }
    }
}