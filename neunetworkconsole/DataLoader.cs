using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace CodingBackProp
{
    /// <summary>Raw data as read from the CSV, before any scaling or splitting.</summary>
    public class RawDataset
    {
        public double[][] Features = new double[0][]; // one row per item, one column per feature
        public int[] Labels = new int[0];             // class index per item, 0-based (0, 1, 2, ...)
        public string[] FeatureNames = new string[0]; // names from the CSV header
        public string[] ClassNames = new string[0];   // original label text, indexed by class index
        public int NumSkipped;        // rows rejected while parsing
    }

    public static class DataLoader
    {
        /// <summary>
        /// Reads a comma-separated file. One column holds the integer class label,
        /// all other columns are numeric features. Rows that have the wrong number
        /// of columns, or a missing / non-numeric value, are skipped and reported.
        /// </summary>
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
                if (lines[i].Trim().Length == 0) continue; // blank line, not a data row

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

            // map raw labels (e.g. 1, 2, 3) to class indexes (0, 1, 2) in sorted order
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