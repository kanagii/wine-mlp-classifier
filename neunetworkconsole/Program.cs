using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;

namespace CodingBackProp
{
    class Program
    {
        // ---- Settings chosen from the tuning grid (see results/experiments.csv) ----
        const int ChosenHidden = 3;
        const double ChosenLearnRate = 0.1;
        const double ChosenMomentum = 0.01;
        const double ChosenWeightDecay = 0.0;
        const int ChosenEpochs = 2000;

        const int SplitSeed = 1;       // fixed seed, so every run uses the same split
        const int LogInterval = 100;   // record the loss every 100 epochs

        static void Main(string[] args)
        {
            Console.WriteLine("\nWine classification with a custom 3-layer MLP");

            // wine.csv is copied next to the .exe by the project file
            string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "wine.csv");
            RawDataset raw = DataLoader.LoadCsv(path, 0, true); // label is column 0
            Console.WriteLine("\nLoaded " + raw.Features.Length + " rows (" +
              raw.NumSkipped + " skipped), " + raw.FeatureNames.Length + " features, " +
              raw.ClassNames.Length + " classes");

            // 60% train, 20% validation, 20% test; features scaled to [0,1] using training rows only
            SplitDataset data = Preprocessor.Prepare(raw, 0.60, 0.20, SplitSeed, true);
            Console.WriteLine("Train " + data.Train.Length + " / validation " +
              data.Validation.Length + " / test " + data.Test.Length + " rows");

            bool running = true;
            while (running)
            {
                Console.WriteLine("\nWhat do you want to run?");
                Console.WriteLine("  1 = train with the chosen settings (train + validation report)");
                Console.WriteLine("  2 = hyperparameter tuning grid (about 1 minute)");
                Console.WriteLine("  3 = scaling experiment (scaled vs unscaled features)");
                Console.WriteLine("  4 = FINAL evaluation on the test set (run once, after tuning)");
                Console.WriteLine("  0 = quit");
                Console.Write("Enter choice: ");
                string choice = (Console.ReadLine() ?? "").Trim();

                if (choice == "1") RunChosen(raw, data);
                else if (choice == "2") RunTuning(data);
                else if (choice == "3") RunScaling(raw);
                else if (choice == "4") RunFinal(raw, data);
                else if (choice == "0") running = false;
                else Console.WriteLine("Please enter 0, 1, 2, 3 or 4.");
            }
            Console.WriteLine("\nEnd of demo\n");
        } // Main

        static TrainingOptions ChosenOptions()
        {
            TrainingOptions opt = new TrainingOptions();
            opt.MaxEpochs = ChosenEpochs;
            opt.LearnRate = ChosenLearnRate;
            opt.Momentum = ChosenMomentum;
            opt.WeightDecay = ChosenWeightDecay;
            opt.LogInterval = LogInterval;
            return opt;
        }

        static void ShowChosenSettings(SplitDataset data)
        {
            Console.WriteLine("\nNetwork: " + data.NumFeatures + "-" + ChosenHidden + "-" + data.NumClasses);
            Console.WriteLine("maxEpochs = " + ChosenEpochs + ", learnRate = " + ChosenLearnRate +
              ", momentum = " + ChosenMomentum + ", weightDecay = " + ChosenWeightDecay);
        }

        // ---- option 1: one training run, reports on train and validation ----
        static void RunChosen(RawDataset raw, SplitDataset data)
        {
            ShowChosenSettings(data);
            NeuralNetwork nn = new NeuralNetwork(data.NumFeatures, ChosenHidden, data.NumClasses);

            Console.WriteLine("\nStarting training");
            List<EpochRecord> history = nn.Train(data.Train, data.Validation, ChosenOptions());
            Console.WriteLine("Done");

            string logPath = Path.Combine(ResultsFolder(), "training_log.csv");
            WriteTrainingLog(logPath, history);
            Console.WriteLine("\nTraining log saved to: " + logPath);

            int[] trainPred = Metrics.Predict(nn, data.Train, data.NumFeatures);
            Metrics.ShowReport("Training set", data.TrainLabels, trainPred, raw.ClassNames);
            int[] valPred = Metrics.Predict(nn, data.Validation, data.NumFeatures);
            Metrics.ShowReport("Validation set", data.ValidationLabels, valPred, raw.ClassNames);
        }

        // ---- option 2: tuning grid ----
        static void RunTuning(SplitDataset data)
        {
            int[] hiddenSizes = { 2, 3, 5, 10, 20 };
            double[] learnRates = { 0.01, 0.05, 0.1, 0.2 };
            double[] weightDecays = { 0.0, 0.0001, 0.001, 0.01 };
            double momentum = 0.01;
            int maxEpochs = 2000;
            int logInterval = 50;

            int total = hiddenSizes.Length * learnRates.Length * weightDecays.Length;
            Console.WriteLine("\nTuning grid: " + total + " runs of up to " + maxEpochs +
              " epochs each (loss checked every " + logInterval + " epochs)");
            Console.WriteLine("Momentum fixed at " + momentum + ". Runs are judged on the VALIDATION set.\n");

            List<ExperimentResult> results = Experiments.RunGrid(data, hiddenSizes, learnRates,
              weightDecays, momentum, maxEpochs, logInterval);

            string csvPath = Path.Combine(ResultsFolder(), "experiments.csv");
            Experiments.WriteCsv(csvPath, results);

            Experiments.SortBestFirst(results);
            Experiments.ShowTop(results, 10);
            Experiments.ShowByHyperparameter(results);
            Console.WriteLine("\nAll " + total + " results saved to: " + csvPath);
        }

        // ---- option 3: scaled vs unscaled ----
        static void RunScaling(RawDataset raw)
        {
            double[] learnRates = { 0.1, 0.01, 0.001, 0.0001, 0.00001 };
            Console.WriteLine("\nScaling experiment: " + raw.FeatureNames.Length + "-" + ChosenHidden + "-" +
              raw.ClassNames.Length + " network, " + ChosenEpochs + " epochs, momentum " + ChosenMomentum);
            Console.WriteLine("Unscaled features get several learning rates, to give them a fair chance.\n");

            string csvPath = Path.Combine(ResultsFolder(), "scaling_experiment.csv");
            Experiments.RunScalingExperiment(raw, ChosenHidden, learnRates, ChosenMomentum,
              ChosenEpochs, LogInterval, SplitSeed, csvPath);
            Console.WriteLine("\nSaved to: " + csvPath);
        }

        // ---- option 4: final test-set evaluation ----
        static void RunFinal(RawDataset raw, SplitDataset data)
        {
            Console.WriteLine("\nThis evaluates on the TEST set. Do it only once, after tuning is finished,");
            Console.WriteLine("otherwise the result is no longer an honest estimate of new-data accuracy.");
            Console.Write("Continue? (y/n): ");
            string answer = (Console.ReadLine() ?? "").Trim().ToLower();
            if (answer != "y") { Console.WriteLine("Cancelled."); return; }

            ShowChosenSettings(data);
            NeuralNetwork nn = new NeuralNetwork(data.NumFeatures, ChosenHidden, data.NumClasses);
            TrainingOptions opt = ChosenOptions();
            opt.Verbose = false;
            nn.Train(data.Train, data.Validation, opt); // test rows are never used for training

            int[] trainPred = Metrics.Predict(nn, data.Train, data.NumFeatures);
            Metrics.ShowReport("Training set", data.TrainLabels, trainPred, raw.ClassNames);
            int[] valPred = Metrics.Predict(nn, data.Validation, data.NumFeatures);
            Metrics.ShowReport("Validation set", data.ValidationLabels, valPred, raw.ClassNames);
            int[] testPred = Metrics.Predict(nn, data.Test, data.NumFeatures);
            Metrics.ShowReport("TEST set (final result)", data.TestLabels, testPred, raw.ClassNames);
        }

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