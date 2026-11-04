using System;
using System.Collections.Generic;

namespace CodingBackProp
{
    public class SplitDataset
    {
        public double[][] Train = new double[0][];
        public double[][] Validation = new double[0][];
        public double[][] Test = new double[0][];
        public int[] TrainLabels = new int[0];
        public int[] ValidationLabels = new int[0];
        public int[] TestLabels = new int[0];
        public double[] FeatureMin = new double[0];
        public double[] FeatureMax = new double[0];
        public int NumFeatures;
        public int NumClasses;
    }

    public static class Preprocessor
    {
        // 1. split into train / validation / test
        // 2. scale every feature to 0-1
        // 3. one-hot encode the class labels
        public static SplitDataset Prepare(RawDataset raw, double trainFrac,
          double valFrac, int seed, bool scale)
        {
            int numFeatures = raw.FeatureNames.Length;
            int numClasses = raw.ClassNames.Length;

            int[][] parts = StratifiedSplit(raw.Labels, numClasses, trainFrac, valFrac, seed);
            int[] trainIdx = parts[0];
            int[] valIdx = parts[1];
            int[] testIdx = parts[2];

            // find the min and max of each feature from the training rows only, so the validation and test data are not used
            double[] min = new double[numFeatures];
            double[] max = new double[numFeatures];
            for (int j = 0; j < numFeatures; ++j)
            {
                min[j] = double.MaxValue;
                max[j] = double.MinValue;
                for (int t = 0; t < trainIdx.Length; ++t)
                {
                    double v = raw.Features[trainIdx[t]][j];
                    if (v < min[j]) min[j] = v;
                    if (v > max[j]) max[j] = v;
                }
            }

            SplitDataset result = new SplitDataset();
            result.NumFeatures = numFeatures;
            result.NumClasses = numClasses;
            result.FeatureMin = min;
            result.FeatureMax = max;
            result.Train = BuildRows(raw, trainIdx, min, max, numClasses, scale);
            result.Validation = BuildRows(raw, valIdx, min, max, numClasses, scale);
            result.Test = BuildRows(raw, testIdx, min, max, numClasses, scale);
            result.TrainLabels = PickLabels(raw.Labels, trainIdx);
            result.ValidationLabels = PickLabels(raw.Labels, valIdx);
            result.TestLabels = PickLabels(raw.Labels, testIdx);
            return result;
        }

        // splits each class separately so train, validation and test all have the same mix of wine types
        private static int[][] StratifiedSplit(int[] labels, int numClasses,
          double trainFrac, double valFrac, int seed)
        {
            Random rnd = new Random(seed);
            List<int> train = new List<int>();
            List<int> val = new List<int>();
            List<int> test = new List<int>();

            for (int c = 0; c < numClasses; ++c)
            {
                List<int> idx = new List<int>();
                for (int i = 0; i < labels.Length; ++i)
                    if (labels[i] == c) idx.Add(i);
                for (int i = 0; i < idx.Count; ++i)
                {
                    int r = rnd.Next(i, idx.Count);
                    int tmp = idx[r]; idx[r] = idx[i]; idx[i] = tmp;
                }

                int n = idx.Count;
                int nTrain = (int)Math.Round(n * trainFrac);
                int nVal = (int)Math.Round(n * valFrac);
                if (nTrain + nVal > n) nVal = n - nTrain;

                for (int i = 0; i < n; ++i)
                {
                    if (i < nTrain) train.Add(idx[i]);
                    else if (i < nTrain + nVal) val.Add(idx[i]);
                    else test.Add(idx[i]);
                }
            }

            Shuffle(train, rnd);
            Shuffle(val, rnd);
            Shuffle(test, rnd);
            return new int[][] { train.ToArray(), val.ToArray(), test.ToArray() };
        }

        private static void Shuffle(List<int> list, Random rnd)
        {
            for (int i = 0; i < list.Count; ++i)
            {
                int r = rnd.Next(i, list.Count);
                int tmp = list[r]; list[r] = list[i]; list[i] = tmp;
            }
        }

        private static double[][] BuildRows(RawDataset raw, int[] idx, double[] min,
          double[] max, int numClasses, bool scale)
        {
            int numFeatures = min.Length;
            double[][] rows = new double[idx.Length][];
            for (int r = 0; r < idx.Length; ++r)
            {
                double[] row = new double[numFeatures + numClasses];
                for (int j = 0; j < numFeatures; ++j)
                {
                    double v = raw.Features[idx[r]][j];
                    if (scale)
                    {
                        double range = max[j] - min[j];
                        // min-max scaling
                        v = (range > 0.0) ? (v - min[j]) / range : 0.0;
                    }
                    row[j] = v;
                }
                row[numFeatures + raw.Labels[idx[r]]] = 1.0;
                rows[r] = row;
            }
            return rows;
        }

        private static int[] PickLabels(int[] labels, int[] idx)
        {
            int[] result = new int[idx.Length];
            for (int i = 0; i < idx.Length; ++i)
                result[i] = labels[idx[i]];
            return result;
        }
    }
}