using System.Globalization;

namespace CodingBackProp
{
    public class WineRunner
    {
        const int ChosenHidden = 2;
        const int ChosenEpochs = 2000;

        const int SplitSeed = 1;
        // show the error every 100 epochs
        const int LogInterval = 100;
        const int Repeats = 10;
        const string ModelFile = "wine_model.txt";
        private RawDataset raw;
        private SplitDataset data;

        public WineRunner()
        {
            ShowIntro();

            string path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "wine.csv");
            raw = DataLoader.LoadCsv(path, 0, true);
            Console.WriteLine("Loaded " + raw.Features.Length + " rows (" +
              raw.NumSkipped + " skipped), " + raw.FeatureNames.Length + " features, " +
              raw.ClassNames.Length + " classes");

            // 60% train, 20% validation, 20% test
            data = Preprocessor.Prepare(raw, 0.60, 0.20, SplitSeed, true);
            Console.WriteLine("Train " + data.Train.Length + " / validation " +
              data.Validation.Length + " / test " + data.Test.Length + " rows");
        }

        private static void ShowIntro()
        {
            Console.WriteLine("WHAT THIS PROGRAM DOES");
            Console.WriteLine("It teaches a small neural network to recognise 3 types of wine from");
            Console.WriteLine("13 chemical measurements (alcohol, color, proline, and so on).");
            Console.WriteLine("The 178 wines are split into three groups:");
            Console.WriteLine("  Training   (107) - the wines the network learns from");
            Console.WriteLine("  Validation  (36) - wines used to compare different settings");
            Console.WriteLine("  Test        (35) - wines kept hidden until the final exam");
            Console.WriteLine("");
            Console.WriteLine("Before training, every measurement is put on the same 0 to 1 scale,");
            Console.WriteLine("so that big numbers (like proline) do not drown out small ones (like hue).");
            Console.WriteLine("");
            Console.WriteLine("THE BUTTONS");
            Console.WriteLine("  1. Train             Teach the network and watch its error fall.");
            Console.WriteLine("  2. Compare settings  Try different network sizes to find the best one.");
            Console.WriteLine("  3. Final test        The one-time exam on the 35 hidden wines. Press");
            Console.WriteLine("                       it once, after buttons 1 and 2.");
            Console.WriteLine("");
            Console.WriteLine("----------------------------------------------------------------");
        }

        public bool ModelExists
        {
            get { return File.Exists(Path.Combine(ResultsFolder(), ModelFile)); }
        }

        private void ShowChosenSettings()
        {
            Console.WriteLine("\nNetwork: " + data.NumFeatures + "-" + ChosenHidden + "-" + data.NumClasses +
              "   epochs = " + ChosenEpochs);
        }

        public void RunChosen()
        {
            ShowChosenSettings();
            MlpClassifier model = new MlpClassifier(data.NumFeatures, ChosenHidden, data.NumClasses);

            TrainingOptions opt = new TrainingOptions();
            opt.MaxEpochs = ChosenEpochs;
            opt.LogInterval = LogInterval;

            Console.WriteLine("\nStarting training");
            // the error is printed while it trains
            List<EpochRecord> history = model.Train(data.Train, data.Validation, opt);
            Console.WriteLine("Done");

            string logPath = Path.Combine(ResultsFolder(), "training_log.csv");
            WriteTrainingLog(logPath, history);
            Console.WriteLine("\nTraining log saved to: " + logPath);

            ShowReports(model, false);
        }

        public void RunTuning()
        {
            int[] hiddenSizes = { 1, 2, 3, 5, 10, 20, 50 };
            int maxEpochs = 2000;
            int logInterval = 100;

            Console.WriteLine("\nTuning: " + hiddenSizes.Length + " hidden-layer sizes, " + Repeats +
              " runs each, up to " + maxEpochs + " epochs (loss checked every " + logInterval + ")");
            Console.WriteLine("Runs are judged on the VALIDATION set. The library starts every network");
            Console.WriteLine("from different random weights, so the runs are averaged.\n");

            List<ExperimentResult> results = Experiments.RunGrid(data, hiddenSizes, Repeats,
              maxEpochs, logInterval);

            string csvPath = Path.Combine(ResultsFolder(), "experiments.csv");
            // save all the results to a csv file
            Experiments.WriteCsv(csvPath, results);

            Experiments.ShowTable(results, hiddenSizes, new int[] { 100, 500, 1000, 2000 });
            Experiments.SortBestFirst(results);
            Experiments.ShowTop(results, 8);
            Console.WriteLine("\nAll results saved to: " + csvPath);
        }

        public void RunFinal()
        {
            string modelPath = Path.Combine(ResultsFolder(), ModelFile);

            ShowChosenSettings();
            MlpClassifier model = new MlpClassifier(data.NumFeatures, ChosenHidden, data.NumClasses);
            TrainingOptions opt = new TrainingOptions();
            opt.MaxEpochs = ChosenEpochs;
            opt.LogInterval = LogInterval;
            opt.Verbose = false;
            // the test data is not used for training
            model.Train(data.Train, data.Validation, opt);

            // save the trained network to a file
            model.SaveModel(modelPath);
            Console.WriteLine("\nThe trained network was saved to: " + modelPath);

            // load it back into a new network to check that the saved file works
            MlpClassifier loaded = new MlpClassifier(data.NumFeatures, ChosenHidden, data.NumClasses);
            loaded.LoadModel(modelPath);
            Console.WriteLine("Evaluating the network re-loaded from that file:");
            ShowReports(loaded, true);
        }

        // prints accuracy, confusion matrix, precision, recall and F1
        private void ShowReports(MlpClassifier model, bool includeTest)
        {
            int[] trainPred = Metrics.Predict(model, data.Train, data.NumFeatures);
            Metrics.ShowReport("Training set", data.TrainLabels, trainPred, raw.ClassNames);
            int[] valPred = Metrics.Predict(model, data.Validation, data.NumFeatures);
            Metrics.ShowReport("Validation set", data.ValidationLabels, valPred, raw.ClassNames);
            if (includeTest)
            {
                int[] testPred = Metrics.Predict(model, data.Test, data.NumFeatures);
                Metrics.ShowReport("TEST set (final result)", data.TestLabels, testPred, raw.ClassNames);
            }
        }

        // gets the results folder inside the project folder, so the files end up in the repo
        private static string ResultsFolder()
        {
            string start = AppDomain.CurrentDomain.BaseDirectory;
            string dir = start;
            for (int up = 0; up < 6; ++up)
            {
                if (Directory.GetFiles(dir, "*.csproj").Length > 0)
                {
                    start = dir;
                    break;
                }
                string parent = Path.GetFullPath(Path.Combine(dir, ".."));
                if (parent == dir) break;
                dir = parent;
            }
            string folder = Path.Combine(start, "results");
            Directory.CreateDirectory(folder);
            return folder;
        }

        // saves the error and accuracy of each logged epoch to a csv file
        private static void WriteTrainingLog(string path, List<EpochRecord> history)
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
    }
}