using System;
using System.Collections.Generic;
using Backprop;

namespace CodingBackProp
{
    public class TrainingOptions
    {
        public int MaxEpochs = 2000;
        public int LogInterval = 100;
        public bool Verbose = true;
        public int ShuffleSeed = 1;
    }

    public class EpochRecord
    {
        public int Epoch;
        public double TrainLoss;
        public double ValLoss;
        public double TrainAcc;
        public double ValAcc;
    }

    public class MlpClassifier
    {
        private NeuralNet net;
        private int numInput;
        private int numHidden;
        private int numOutput;

        public MlpClassifier(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;
            this.net = new NeuralNet(numInput, numHidden, numOutput);
        }

        public double[] ComputeOutputs(double[] xValues)
        {
            for (int i = 0; i < numInput; ++i)
                net.setInputs(i, xValues[i]);
            net.run();
            double[] result = new double[numOutput];
            for (int k = 0; k < numOutput; ++k)
                result[k] = net.getOuputData(k);
            return result;
        }

        // trains the network: go through all the training rows again
        public List<EpochRecord> Train(double[][] trainData, double[][] valData,
          TrainingOptions opt)
        {
            List<EpochRecord> history = new List<EpochRecord>();
            Random shuffleRnd = new Random(opt.ShuffleSeed);
            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            for (int epoch = 1; epoch <= opt.MaxEpochs; ++epoch)
            {
                Shuffle(sequence, shuffleRnd);
                for (int ii = 0; ii < sequence.Length; ++ii)
                {
                    // give the network one wine and the correct answer then let it learn
                    double[] row = trainData[sequence[ii]];
                    for (int i = 0; i < numInput; ++i)
                        net.setInputs(i, row[i]);
                    for (int k = 0; k < numOutput; ++k)
                        net.setDesiredOutput(k, row[numInput + k]);
                    net.learn();
                }

                // save the error and accuracy every now and then
                if (epoch == 1 || epoch % opt.LogInterval == 0 || epoch == opt.MaxEpochs)
                {
                    EpochRecord rec = new EpochRecord();
                    rec.Epoch = epoch;
                    rec.TrainLoss = Error(trainData);
                    rec.TrainAcc = Accuracy(trainData);
                    if (valData != null && valData.Length > 0)
                    {
                        rec.ValLoss = Error(valData);
                        rec.ValAcc = Accuracy(valData);
                    }
                    history.Add(rec);
                    if (opt.Verbose)
                        Console.WriteLine("epoch " + epoch.ToString().PadLeft(5) +
                          " | train loss " + rec.TrainLoss.ToString("F4") +
                          " | val loss " + rec.ValLoss.ToString("F4") +
                          " | train acc " + rec.TrainAcc.ToString("F4") +
                          " | val acc " + rec.ValAcc.ToString("F4"));
                }
            }
            return history;
        }

        // shuffles row order
        private static void Shuffle(int[] sequence, Random rnd)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r]; sequence[r] = sequence[i]; sequence[i] = tmp;
            }
        }

        public double Error(double[][] data)
        {
            double sum = 0.0;
            double[] xValues = new double[numInput];
            for (int i = 0; i < data.Length; ++i)
            {
                Array.Copy(data[i], xValues, numInput);
                double[] y = ComputeOutputs(xValues);
                for (int k = 0; k < numOutput; ++k)
                {
                    double diff = data[i][numInput + k] - y[k];
                    sum += diff * diff;
                }
            }
            return sum / data.Length;
        }

        // wines predicted correctly
        public double Accuracy(double[][] data)
        {
            int correct = 0;
            double[] xValues = new double[numInput];
            double[] tValues = new double[numOutput];
            for (int i = 0; i < data.Length; ++i)
            {
                Array.Copy(data[i], xValues, numInput);
                Array.Copy(data[i], numInput, tValues, 0, numOutput);
                if (Metrics.MaxIndex(ComputeOutputs(xValues)) == Metrics.MaxIndex(tValues))
                    ++correct;
            }
            return (correct * 1.0) / data.Length;
        }

        // saves the weights to a file (uses saveWeights from the library)
        public void SaveModel(string path) { net.saveWeights(path); }

        // loads the weights from a file (uses loadWeights), the network needs the same sizes
        public void LoadModel(string path) { net.loadWeights(path); }
    }
}