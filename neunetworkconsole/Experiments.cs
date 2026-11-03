using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace CodingBackProp
{
  /// <summary>Outcome of one training run with one set of hyperparameters.</summary>
  public class ExperimentResult
  {
    public int Hidden;
    public double LearnRate;
    public double Momentum;
    public double WeightDecay;
    public int BestEpoch;          // logged epoch with the lowest validation loss
    public double BestValLoss;
    public double BestValAcc;
    public double TrainLossAtBest;
    public double TrainAccAtBest;
    public double FinalValLoss;    // after the last epoch
    public double FinalValAcc;
    public double FinalTrainAcc;
  }

  public static class Experiments
  {
    /// <summary>Trains one network and summarizes its recorded history.</summary>
    public static ExperimentResult RunConfig(SplitDataset data, int numHidden,
      TrainingOptions opt)
    {
      NeuralNetwork nn = new NeuralNetwork(data.NumFeatures, numHidden, data.NumClasses);
      List<EpochRecord> history = nn.Train(data.Train, data.Validation, opt);

      EpochRecord best = history[0];
      for (int i = 1; i < history.Count; ++i)
      {
        double v = history[i].ValLoss;
        if (!double.IsNaN(v) && (double.IsNaN(best.ValLoss) || v < best.ValLoss))
          best = history[i];
      }
      EpochRecord last = history[history.Count - 1];

      ExperimentResult r = new ExperimentResult();
      r.Hidden = numHidden;
      r.LearnRate = opt.LearnRate;
      r.Momentum = opt.Momentum;
      r.WeightDecay = opt.WeightDecay;
      r.BestEpoch = best.Epoch;
      r.BestValLoss = best.ValLoss;
      r.BestValAcc = best.ValAcc;
      r.TrainLossAtBest = best.TrainLoss;
      r.TrainAccAtBest = best.TrainAcc;
      r.FinalValLoss = last.ValLoss;
      r.FinalValAcc = last.ValAcc;
      r.FinalTrainAcc = last.TrainAcc;
      return r;
    }

    /// <summary>Tries every combination of the given hyperparameter values.</summary>
    public static List<ExperimentResult> RunGrid(SplitDataset data, int[] hiddenSizes,
      double[] learnRates, double[] weightDecays, double momentum, int maxEpochs,
      int logInterval)
    {
      List<ExperimentResult> results = new List<ExperimentResult>();
      int total = hiddenSizes.Length * learnRates.Length * weightDecays.Length;
      int run = 0;

      Console.WriteLine("  run  hidden  learnRate  decay     bestEpoch  valLoss  valAcc  trainLoss");
      foreach (int h in hiddenSizes)
      {
        foreach (double lr in learnRates)
        {
          foreach (double wd in weightDecays)
          {
            ++run;
            TrainingOptions opt = new TrainingOptions();
            opt.MaxEpochs = maxEpochs;
            opt.LearnRate = lr;
            opt.Momentum = momentum;
            opt.WeightDecay = wd;
            opt.LogInterval = logInterval;
            opt.Verbose = false;

            ExperimentResult r = RunConfig(data, h, opt);
            results.Add(r);
            Console.WriteLine("  " + (run + "/" + total).PadRight(6) +
              h.ToString().PadLeft(6) + lr.ToString("0.###").PadLeft(11) +
              wd.ToString("0.####").PadLeft(8) + r.BestEpoch.ToString().PadLeft(11) +
              r.BestValLoss.ToString("F4").PadLeft(9) + r.BestValAcc.ToString("F4").PadLeft(8) +
              r.TrainLossAtBest.ToString("F4").PadLeft(10));
          }
        }
      }
      return results;
    }

    private static double SortKey(ExperimentResult r)
    {
      return double.IsNaN(r.BestValLoss) ? double.MaxValue : r.BestValLoss;
    }

    /// <summary>Sorts best-first: lowest validation loss, then highest validation accuracy.</summary>
    public static void SortBestFirst(List<ExperimentResult> results)
    {
      results.Sort(delegate (ExperimentResult a, ExperimentResult b)
      {
        int c = SortKey(a).CompareTo(SortKey(b));
        if (c != 0) return c;
        return b.BestValAcc.CompareTo(a.BestValAcc);
      });
    }

    public static void ShowTop(List<ExperimentResult> sorted, int count)
    {
      Console.WriteLine("\nTop " + count + " configurations (lowest validation loss):");
      Console.WriteLine("  rank  hidden  learnRate  decay     epochs  valLoss  valAcc  trainLoss");
      for (int i = 0; i < count && i < sorted.Count; ++i)
      {
        ExperimentResult r = sorted[i];
        Console.WriteLine("  " + (i + 1).ToString().PadLeft(4) + r.Hidden.ToString().PadLeft(8) +
          r.LearnRate.ToString("0.###").PadLeft(11) + r.WeightDecay.ToString("0.####").PadLeft(8) +
          r.BestEpoch.ToString().PadLeft(9) + r.BestValLoss.ToString("F4").PadLeft(9) +
          r.BestValAcc.ToString("F4").PadLeft(8) + r.TrainLossAtBest.ToString("F4").PadLeft(10));
      }
    }

    /// <summary>
    /// For each value of each hyperparameter, shows the best result achieved with
    /// that value (best over all the other settings). Handy for the write-up.
    /// </summary>
    public static void ShowByHyperparameter(List<ExperimentResult> results)
    {
      Console.WriteLine("\nBest validation loss reached with each value (over all other settings):");

      SortedSet<int> hs = new SortedSet<int>();
      SortedSet<double> lrs = new SortedSet<double>();
      SortedSet<double> wds = new SortedSet<double>();
      foreach (ExperimentResult r in results)
      {
        hs.Add(r.Hidden); lrs.Add(r.LearnRate); wds.Add(r.WeightDecay);
      }

      Console.WriteLine("  hidden nodes:");
      foreach (int h in hs)
        PrintBestWhere(results, delegate (ExperimentResult r) { return r.Hidden == h; }, h.ToString());
      Console.WriteLine("  learning rate:");
      foreach (double lr in lrs)
        PrintBestWhere(results, delegate (ExperimentResult r) { return r.LearnRate == lr; }, lr.ToString("0.###"));
      Console.WriteLine("  weight decay:");
      foreach (double wd in wds)
        PrintBestWhere(results, delegate (ExperimentResult r) { return r.WeightDecay == wd; }, wd.ToString("0.####"));
    }

    private static void PrintBestWhere(List<ExperimentResult> results,
      Predicate<ExperimentResult> match, string label)
    {
      int bestIdx = -1;
      for (int i = 0; i < results.Count; ++i)
      {
        if (!match(results[i])) continue;
        if (bestIdx < 0 || SortKey(results[i]) < SortKey(results[bestIdx])) bestIdx = i;
      }
      if (bestIdx >= 0)
      {
        ExperimentResult best = results[bestIdx];
        Console.WriteLine("    " + label.PadRight(8) + " valLoss " + best.BestValLoss.ToString("F4") +
          "  valAcc " + best.BestValAcc.ToString("F4") + "  (epoch " + best.BestEpoch + ")");
      }
    }

    public static void WriteCsv(string path, List<ExperimentResult> results)
    {
      List<string> lines = new List<string>();
      lines.Add("hidden,learn_rate,momentum,weight_decay,best_epoch,best_val_loss," +
        "best_val_acc,train_loss_at_best,train_acc_at_best,final_val_loss,final_val_acc,final_train_acc");
      CultureInfo ci = CultureInfo.InvariantCulture;
      foreach (ExperimentResult r in results)
      {
        lines.Add(r.Hidden + "," + r.LearnRate.ToString(ci) + "," + r.Momentum.ToString(ci) + "," +
          r.WeightDecay.ToString(ci) + "," + r.BestEpoch + "," +
          r.BestValLoss.ToString("F6", ci) + "," + r.BestValAcc.ToString("F6", ci) + "," +
          r.TrainLossAtBest.ToString("F6", ci) + "," + r.TrainAccAtBest.ToString("F6", ci) + "," +
          r.FinalValLoss.ToString("F6", ci) + "," + r.FinalValAcc.ToString("F6", ci) + "," +
          r.FinalTrainAcc.ToString("F6", ci));
      }
      File.WriteAllLines(path, lines);
    }

    /// <summary>
    /// Trains the same network on scaled and on unscaled features, at several
    /// learning rates, to show what feature scaling does. Everything else
    /// (split, seed, network size, epochs) is identical between the two.
    /// </summary>
    public static void RunScalingExperiment(RawDataset raw, int numHidden,
      double[] learnRates, double momentum, int maxEpochs, int logInterval,
      int splitSeed, string csvPath)
    {
      List<string> lines = new List<string>();
      lines.Add("scaled,learn_rate,best_epoch,best_val_loss,final_train_acc,final_val_acc");
      CultureInfo ci = CultureInfo.InvariantCulture;

      Console.WriteLine("  features   learnRate  bestEpoch  bestValLoss  finalTrainAcc  finalValAcc");
      foreach (bool scale in new bool[] { true, false })
      {
        SplitDataset data = Preprocessor.Prepare(raw, 0.60, 0.20, splitSeed, scale);
        foreach (double lr in learnRates)
        {
          TrainingOptions opt = new TrainingOptions();
          opt.MaxEpochs = maxEpochs;
          opt.LearnRate = lr;
          opt.Momentum = momentum;
          opt.WeightDecay = 0.0;
          opt.LogInterval = logInterval;
          opt.Verbose = false;
          ExperimentResult r = RunConfig(data, numHidden, opt);

          Console.WriteLine("  " + (scale ? "scaled" : "unscaled").PadRight(10) +
            lr.ToString("0.#####").PadLeft(10) + r.BestEpoch.ToString().PadLeft(11) +
            r.BestValLoss.ToString("F4").PadLeft(13) + r.FinalTrainAcc.ToString("F4").PadLeft(15) +
            r.FinalValAcc.ToString("F4").PadLeft(13));
          lines.Add((scale ? "scaled" : "unscaled") + "," + lr.ToString(ci) + "," +
            r.BestEpoch + "," + r.BestValLoss.ToString("F6", ci) + "," +
            r.FinalTrainAcc.ToString("F6", ci) + "," + r.FinalValAcc.ToString("F6", ci));
        }
      }
      File.WriteAllLines(csvPath, lines);
    }
  }
}
