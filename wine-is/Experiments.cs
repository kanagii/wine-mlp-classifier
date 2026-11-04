using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace CodingBackProp
{
    public class ExperimentResult
    {
        public int Hidden;
        public int Repeats;
        public int Epoch;
        public double TrainLoss;
        public double ValLoss;
        public double TrainAcc;
        public double ValAcc;
        public double ValAccMin;
        public double ValAccMax;
    }

    public static class Experiments
    {
        // trains the same network several times and averages the results
        public static List<ExperimentResult> RunRepeated(SplitDataset data, int numHidden,
          int repeats, int maxEpochs, int logInterval)
        {
            List<List<EpochRecord>> histories = new List<List<EpochRecord>>();
            for (int r = 0; r < repeats; ++r)
            {
                MlpClassifier model = new MlpClassifier(data.NumFeatures, numHidden, data.NumClasses);
                TrainingOptions opt = new TrainingOptions();
                opt.MaxEpochs = maxEpochs;
                opt.LogInterval = logInterval;
                opt.Verbose = false;
                opt.ShuffleSeed = r + 1;
                histories.Add(model.Train(data.Train, data.Validation, opt));
            }

            List<ExperimentResult> results = new List<ExperimentResult>();
            int points = histories[0].Count;
            for (int p = 0; p < points; ++p)
            {
                ExperimentResult res = new ExperimentResult();
                res.Hidden = numHidden;
                res.Repeats = repeats;
                res.Epoch = histories[0][p].Epoch;
                res.ValAccMin = double.MaxValue;
                res.ValAccMax = double.MinValue;
                for (int r = 0; r < repeats; ++r)
                {
                    EpochRecord rec = histories[r][p];
                    res.TrainLoss += rec.TrainLoss / repeats;
                    res.ValLoss += rec.ValLoss / repeats;
                    res.TrainAcc += rec.TrainAcc / repeats;
                    res.ValAcc += rec.ValAcc / repeats;
                    if (rec.ValAcc < res.ValAccMin) res.ValAccMin = rec.ValAcc;
                    if (rec.ValAcc > res.ValAccMax) res.ValAccMax = rec.ValAcc;
                }
                results.Add(res);
            }
            return results;
        }

        // tries each hidden layer size
        public static List<ExperimentResult> RunGrid(SplitDataset data, int[] hiddenSizes,
          int repeats, int maxEpochs, int logInterval)
        {
            List<ExperimentResult> all = new List<ExperimentResult>();
            for (int i = 0; i < hiddenSizes.Length; ++i)
            {
                Console.WriteLine("  hidden nodes = " + hiddenSizes[i] + "  (" + repeats +
                  " runs)   [" + (i + 1) + "/" + hiddenSizes.Length + "]");
                all.AddRange(RunRepeated(data, hiddenSizes[i], repeats, maxEpochs, logInterval));
            }
            return all;
        }

        // finds the result for one hidden size and epoch
        private static int FindIndex(List<ExperimentResult> results, int hidden, int epoch)
        {
            for (int i = 0; i < results.Count; ++i)
                if (results[i].Hidden == hidden && results[i].Epoch == epoch) return i;
            return -1;
        }

        // prints the average validation error and accuracy for each hidden size
        public static void ShowTable(List<ExperimentResult> results, int[] hiddenSizes, int[] epochs)
        {
            Console.WriteLine("\nMean validation loss / accuracy (average over the repeated runs):");
            string header = "  hidden";
            foreach (int e in epochs) header += ("epoch " + e).PadLeft(18);
            Console.WriteLine(header);
            foreach (int h in hiddenSizes)
            {
                string line = "  " + h.ToString().PadRight(6);
                foreach (int e in epochs)
                {
                    int idx = FindIndex(results, h, e);
                    line += (idx < 0) ? "n/a".PadLeft(18) :
                      (results[idx].ValLoss.ToString("F4") + " / " + results[idx].ValAcc.ToString("F4")).PadLeft(18);
                }
                Console.WriteLine(line);
            }
        }

        // puts the best result first
        public static void SortBestFirst(List<ExperimentResult> results)
        {
            results.Sort(delegate (ExperimentResult a, ExperimentResult b)
            {
                int c = a.ValLoss.CompareTo(b.ValLoss);
                if (c != 0) return c;
                return b.ValAcc.CompareTo(a.ValAcc);
            });
        }

        // prints the best results
        public static void ShowTop(List<ExperimentResult> sorted, int count)
        {
            Console.WriteLine("\nTop " + count + " (hidden, epochs) settings by mean validation loss:");
            Console.WriteLine("  rank  hidden  epochs  valLoss  valAcc (min-max over runs)  trainLoss  trainAcc");
            for (int i = 0; i < count && i < sorted.Count; ++i)
            {
                ExperimentResult r = sorted[i];
                Console.WriteLine("  " + (i + 1).ToString().PadLeft(4) + r.Hidden.ToString().PadLeft(8) +
                  r.Epoch.ToString().PadLeft(8) + r.ValLoss.ToString("F4").PadLeft(9) +
                  (r.ValAcc.ToString("F4") + " (" + r.ValAccMin.ToString("F3") + "-" +
                  r.ValAccMax.ToString("F3") + ")").PadLeft(28) +
                  r.TrainLoss.ToString("F4").PadLeft(11) + r.TrainAcc.ToString("F4").PadLeft(10));
            }
        }

        // saves all the results to a csv file
        public static void WriteCsv(string path, List<ExperimentResult> results)
        {
            List<string> lines = new List<string>();
            lines.Add("hidden,repeats,epoch,train_loss,val_loss,train_acc,val_acc,val_acc_min,val_acc_max");
            CultureInfo ci = CultureInfo.InvariantCulture;
            foreach (ExperimentResult r in results)
            {
                lines.Add(r.Hidden + "," + r.Repeats + "," + r.Epoch + "," +
                  r.TrainLoss.ToString("F6", ci) + "," + r.ValLoss.ToString("F6", ci) + "," +
                  r.TrainAcc.ToString("F6", ci) + "," + r.ValAcc.ToString("F6", ci) + "," +
                  r.ValAccMin.ToString("F6", ci) + "," + r.ValAccMax.ToString("F6", ci));
            }
            File.WriteAllLines(path, lines);
        }
    }
}