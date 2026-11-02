using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace CodingBackProp
{
    class Program
    {
        // The test set should be looked at ONCE, after all tuning is finished.
        // Leave this false while experimenting; set it to true for the final run.
        static readonly bool EvaluateOnTest = false;

        static void Main(string[] args)
        {
            Console.WriteLine("\nWine classification with a custom 3-layer MLP");

            // ---- 1. load the data (wine.csv is copied next to the .exe by the project file)
            string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "wine.csv");
            RawDataset raw = DataLoader.LoadCsv(path, 0, true); // label is column 0
            Console.WriteLine("\nLoaded " + raw.Features.Length + " rows (" +
              raw.NumSkipped + " skipped), " + raw.FeatureNames.Length + " features, " +
              raw.ClassNames.Length + " classes");

            // ---- 2. split 60/20/20, scale features to [0,1] using training rows only
            SplitDataset data = Preprocessor.Prepare(raw, 0.60, 0.20, 1, true);
            Console.WriteLine("Train " + data.Train.Length + " / validation " +
              data.Validation.Length + " / test " + data.Test.Length + " rows");

            // ---- 3. choose settings and build the network
            int numHidden = 5;
            TrainingOptions opt = new TrainingOptions();
            opt.MaxEpochs = 1000;
            opt.LearnRate = 0.05;
            opt.Momentum = 0.01;
            opt.WeightDecay = 0.0;
            opt.LogInterval = 100;

            Console.WriteLine("\nNetwork: " + data.NumFeatures + "-" + numHidden + "-" +
              data.NumClasses);
            Console.WriteLine("maxEpochs = " + opt.MaxEpochs + ", learnRate = " + opt.LearnRate +
              ", momentum = " + opt.Momentum + ", weightDecay = " + opt.WeightDecay);

            NeuralNetwork nn = new NeuralNetwork(data.NumFeatures, numHidden, data.NumClasses);

            // ---- 4. train, recording loss every LogInterval epochs
            Console.WriteLine("\nStarting training");
            List<EpochRecord> history = nn.Train(data.Train, data.Validation, opt);
            Console.WriteLine("Done");

            string logPath = Path.Combine(ResultsFolder(), "training_log.csv");
            WriteTrainingLog(logPath, history);
            Console.WriteLine("\nTraining log saved to: " + logPath);

            // ---- 5. evaluate
            int[] trainPred = Metrics.Predict(nn, data.Train, data.NumFeatures);
            Metrics.ShowReport("Training set", data.TrainLabels, trainPred, raw.ClassNames);

            int[] valPred = Metrics.Predict(nn, data.Validation, data.NumFeatures);
            Metrics.ShowReport("Validation set", data.ValidationLabels, valPred, raw.ClassNames);

            if (EvaluateOnTest)
            {
                int[] testPred = Metrics.Predict(nn, data.Test, data.NumFeatures);
                Metrics.ShowReport("Test set (final)", data.TestLabels, testPred, raw.ClassNames);
            }
            else
            {
                Console.WriteLine("\n(Test set not evaluated yet. Set EvaluateOnTest = true for the final run.)");
            }

            Console.WriteLine("\nEnd of demo\n");
            Console.ReadLine();
        } // Main

        /// <summary>
        /// Folder named "results" inside the project folder (the one holding the .csproj),
        /// so saved files end up in the repo rather than in bin/Debug.
        /// Falls back to the .exe folder if the project folder can't be found.
        /// </summary>
        static string ResultsFolder()
        {
            string start = AppDomain.CurrentDomain.BaseDirectory;
            string dir = start;
            for (int up = 0; up < 6; ++up)
            {
                if (Directory.GetFiles(dir, "*.csproj").Length > 0)
                {
                    start = dir; // found the project folder
                    break;
                }
                string parent = Path.GetFullPath(Path.Combine(dir, ".."));
                if (parent == dir) break; // reached the drive root
                dir = parent;
            }
            string folder = Path.Combine(start, "results");
            Directory.CreateDirectory(folder);
            return folder;
        }

        static void WriteTrainingLog(string path, List<EpochRecord> history)
        {
            List<string> lines = new List<string>();
            lines.Add("epoch,train_loss,val_loss,train_acc,val_acc");
            for (int i = 0; i < history.Count; ++i)
            {
                EpochRecord r = history[i];
                lines.Add(r.Epoch + "," +
                  r.TrainLoss.ToString("F6", CultureInfo.InvariantCulture) + "," +
                  r.ValLoss.ToString("F6", CultureInfo.InvariantCulture) + "," +
                  r.TrainAcc.ToString("F6", CultureInfo.InvariantCulture) + "," +
                  r.ValAcc.ToString("F6", CultureInfo.InvariantCulture));
            }
            File.WriteAllLines(path, lines);
        }
    } // Program
} // ns