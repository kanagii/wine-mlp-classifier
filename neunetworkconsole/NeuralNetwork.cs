using System;

namespace CodingBackProp
{
  /// <summary>
  /// A fully connected 3-layer neural network (input, one hidden layer, output)
  /// trained with back-propagation. Hidden layer uses tanh, output layer uses
  /// softmax, and training minimizes cross-entropy error.
  /// </summary>
  public class NeuralNetwork
  {
    private int numInput;   // number input nodes
    private int numHidden;
    private int numOutput;

    private double[] inputs;
    private double[][] ihWeights; // input-hidden
    private double[] hBiases;
    private double[] hOutputs;

    private double[][] hoWeights; // hidden-output
    private double[] oBiases;
    private double[] outputs;

    private Random rnd;

    public NeuralNetwork(int numInput, int numHidden, int numOutput)
    {
      this.numInput = numInput;
      this.numHidden = numHidden;
      this.numOutput = numOutput;

      this.inputs = new double[numInput];

      this.ihWeights = MakeMatrix(numInput, numHidden, 0.0);
      this.hBiases = new double[numHidden];
      this.hOutputs = new double[numHidden];

      this.hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
      this.oBiases = new double[numOutput];
      this.outputs = new double[numOutput];

      this.rnd = new Random(0);
      this.InitializeWeights(); // all weights and biases
    } // ctor

    private static double[][] MakeMatrix(int rows, int cols, double v)
    {
      double[][] result = new double[rows][];
      for (int r = 0; r < result.Length; ++r)
        result[r] = new double[cols];
      for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
          result[i][j] = v;
      return result;
    }

    private void InitializeWeights()
    {
      // initialize weights and biases to small random values
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      double[] initialWeights = new double[numWeights];
      for (int i = 0; i < initialWeights.Length; ++i)
        initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
      this.SetWeights(initialWeights);
    }

    public void SetWeights(double[] weights)
    {
      // copy serialized weights and biases in weights[] array
      // to i-h weights, i-h biases, h-o weights, h-o biases
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      if (weights.Length != numWeights)
        throw new Exception("Bad weights array in SetWeights");

      int k = 0; // points into weights param

      for (int i = 0; i < numInput; ++i)
        for (int j = 0; j < numHidden; ++j)
          ihWeights[i][j] = weights[k++];
      for (int i = 0; i < numHidden; ++i)
        hBiases[i] = weights[k++];
      for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
          hoWeights[i][j] = weights[k++];
      for (int i = 0; i < numOutput; ++i)
        oBiases[i] = weights[k++];
    }

    public double[] GetWeights()
    {
      int numWeights = (numInput * numHidden) +
        (numHidden * numOutput) + numHidden + numOutput;
      double[] result = new double[numWeights];
      int k = 0;
      for (int i = 0; i < ihWeights.Length; ++i)
        for (int j = 0; j < ihWeights[0].Length; ++j)
          result[k++] = ihWeights[i][j];
      for (int i = 0; i < hBiases.Length; ++i)
        result[k++] = hBiases[i];
      for (int i = 0; i < hoWeights.Length; ++i)
        for (int j = 0; j < hoWeights[0].Length; ++j)
          result[k++] = hoWeights[i][j];
      for (int i = 0; i < oBiases.Length; ++i)
        result[k++] = oBiases[i];
      return result;
    }

    /// <summary>Forward pass: inputs -> tanh hidden layer -> softmax outputs.</summary>
    public double[] ComputeOutputs(double[] xValues)
    {
      double[] hSums = new double[numHidden]; // hidden nodes sums scratch array
      double[] oSums = new double[numOutput]; // output nodes sums

      for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
        this.inputs[i] = xValues[i];

      for (int j = 0; j < numHidden; ++j)  // sum of weights * inputs
        for (int i = 0; i < numInput; ++i)
          hSums[j] += this.inputs[i] * this.ihWeights[i][j];

      for (int i = 0; i < numHidden; ++i)  // add biases to hidden sums
        hSums[i] += this.hBiases[i];

      for (int i = 0; i < numHidden; ++i)   // apply activation
        this.hOutputs[i] = HyperTan(hSums[i]);

      for (int j = 0; j < numOutput; ++j)   // sum of weights * hidden outputs
        for (int i = 0; i < numHidden; ++i)
          oSums[j] += hOutputs[i] * hoWeights[i][j];

      for (int i = 0; i < numOutput; ++i)  // add biases to output sums
        oSums[i] += oBiases[i];

      double[] softOut = Softmax(oSums);
      Array.Copy(softOut, outputs, softOut.Length);

      double[] retResult = new double[numOutput];
      Array.Copy(this.outputs, retResult, retResult.Length);
      return retResult;
    }

    private static double HyperTan(double x)
    {
      if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
      else if (x > 20.0) return 1.0;
      else return Math.Tanh(x);
    }

    private static double[] Softmax(double[] oSums)
    {
      // does all output nodes at once so scale doesn't have to be re-computed
      double sum = 0.0;
      for (int i = 0; i < oSums.Length; ++i)
        sum += Math.Exp(oSums[i]);

      double[] result = new double[oSums.Length];
      for (int i = 0; i < oSums.Length; ++i)
        result[i] = Math.Exp(oSums[i]) / sum;

      return result; // now scaled so that values sum to 1.0
    }

    /// <summary>
    /// Trains with online (stochastic) back-propagation.
    /// Each row of trainData is: input values followed by 1-of-N target values.
    /// </summary>
    public double[] Train(double[][] trainData, int maxEpochs,
      double learnRate, double momentum, double weightDecay)
    {
      // gradients
      double[][] hoGrads = MakeMatrix(numHidden, numOutput, 0.0); // hidden-to-output weight gradients
      double[] obGrads = new double[numOutput];                   // output bias gradients
      double[][] ihGrads = MakeMatrix(numInput, numHidden, 0.0);  // input-to-hidden weight gradients
      double[] hbGrads = new double[numHidden];                   // hidden bias gradients

      // local gradient signals (gradients without the associated input term)
      double[] oSignals = new double[numOutput];
      double[] hSignals = new double[numHidden];

      // previous deltas, needed for momentum
      double[][] ihPrevWeightsDelta = MakeMatrix(numInput, numHidden, 0.0);
      double[] hPrevBiasesDelta = new double[numHidden];
      double[][] hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput, 0.0);
      double[] oPrevBiasesDelta = new double[numOutput];

      int epoch = 0;
      double[] xValues = new double[numInput]; // inputs
      double[] tValues = new double[numOutput]; // target values
      double derivative = 0.0;

      int[] sequence = new int[trainData.Length];
      for (int i = 0; i < sequence.Length; ++i)
        sequence[i] = i;

      int errInterval = maxEpochs / 10; // interval to check error
      while (epoch < maxEpochs)
      {
        ++epoch;

        if (epoch % errInterval == 0 && epoch < maxEpochs)
        {
          double trainErr = Error(trainData);
          Console.WriteLine("epoch = " + epoch + "  error = " +
            trainErr.ToString("F4"));
        }

        Shuffle(sequence); // visit each training item in random order
        for (int ii = 0; ii < trainData.Length; ++ii)
        {
          int idx = sequence[ii];
          Array.Copy(trainData[idx], xValues, numInput);
          Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
          ComputeOutputs(xValues); // forward pass

          // indices: i = inputs, j = hiddens, k = outputs

          // 1. output node signals (cross-entropy + softmax: target - output)
          for (int k = 0; k < numOutput; ++k)
            oSignals[k] = tValues[k] - outputs[k];

          // 2. hidden-to-output weight gradients
          for (int j = 0; j < numHidden; ++j)
            for (int k = 0; k < numOutput; ++k)
              hoGrads[j][k] = oSignals[k] * hOutputs[j];

          // 2b. output bias gradients
          for (int k = 0; k < numOutput; ++k)
            obGrads[k] = oSignals[k] * 1.0; // dummy associated input value

          // 3. hidden node signals (the "backward" step)
          for (int j = 0; j < numHidden; ++j)
          {
            derivative = (1 + hOutputs[j]) * (1 - hOutputs[j]); // tanh derivative
            double sum = 0.0;
            for (int k = 0; k < numOutput; ++k)
              sum += oSignals[k] * hoWeights[j][k];
            hSignals[j] = derivative * sum;
          }

          // 4. input-to-hidden weight gradients
          for (int i = 0; i < numInput; ++i)
            for (int j = 0; j < numHidden; ++j)
              ihGrads[i][j] = hSignals[j] * inputs[i];

          // 4b. hidden bias gradients
          for (int j = 0; j < numHidden; ++j)
            hbGrads[j] = hSignals[j] * 1.0; // dummy 1.0 input

          // == update weights and biases ==
          // (weight decay is applied to weights only, not to biases)

          // input-to-hidden weights
          for (int i = 0; i < numInput; ++i)
          {
            for (int j = 0; j < numHidden; ++j)
            {
              double delta = ihGrads[i][j] * learnRate;
              ihWeights[i][j] += delta;
              ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
              ihWeights[i][j] -= learnRate * weightDecay * ihWeights[i][j];
              ihPrevWeightsDelta[i][j] = delta; // save for momentum
            }
          }

          // hidden biases
          for (int j = 0; j < numHidden; ++j)
          {
            double delta = hbGrads[j] * learnRate;
            hBiases[j] += delta;
            hBiases[j] += hPrevBiasesDelta[j] * momentum;
            hPrevBiasesDelta[j] = delta;
          }

          // hidden-to-output weights
          for (int j = 0; j < numHidden; ++j)
          {
            for (int k = 0; k < numOutput; ++k)
            {
              double delta = hoGrads[j][k] * learnRate;
              hoWeights[j][k] += delta;
              hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
              hoWeights[j][k] -= learnRate * weightDecay * hoWeights[j][k];
              hoPrevWeightsDelta[j][k] = delta;
            }
          }

          // output biases
          for (int k = 0; k < numOutput; ++k)
          {
            double delta = obGrads[k] * learnRate;
            oBiases[k] += delta;
            oBiases[k] += oPrevBiasesDelta[k] * momentum;
            oPrevBiasesDelta[k] = delta;
          }
        } // each training item
      } // while

      return GetWeights();
    } // Train

    private void Shuffle(int[] sequence) // Fisher-Yates
    {
      for (int i = 0; i < sequence.Length; ++i)
      {
        int r = this.rnd.Next(i, sequence.Length);
        int tmp = sequence[r];
        sequence[r] = sequence[i];
        sequence[i] = tmp;
      }
    }

    /// <summary>Average cross-entropy error per data item.</summary>
    public double Error(double[][] data)
    {
      double sumError = 0.0;
      double[] xValues = new double[numInput];
      double[] tValues = new double[numOutput];

      for (int i = 0; i < data.Length; ++i)
      {
        Array.Copy(data[i], xValues, numInput);
        Array.Copy(data[i], numInput, tValues, 0, numOutput);
        double[] yValues = this.ComputeOutputs(xValues);
        for (int j = 0; j < numOutput; ++j)
        {
          if (tValues[j] > 0.0) // skip zeros so we never compute 0 * log(0)
            sumError -= tValues[j] * Math.Log(yValues[j]);
        }
      }
      return sumError / data.Length;
    }

    /// <summary>Fraction of items where the largest output matches the target.</summary>
    public double Accuracy(double[][] data)
    {
      int numCorrect = 0;
      double[] xValues = new double[numInput];
      double[] tValues = new double[numOutput];

      for (int i = 0; i < data.Length; ++i)
      {
        Array.Copy(data[i], xValues, numInput);
        Array.Copy(data[i], numInput, tValues, 0, numOutput);
        double[] yValues = this.ComputeOutputs(xValues);
        if (MaxIndex(yValues) == MaxIndex(tValues))
          ++numCorrect;
      }
      return (numCorrect * 1.0) / data.Length;
    }

    public static int MaxIndex(double[] vector)
    {
      int bigIndex = 0;
      double biggestVal = vector[0];
      for (int i = 0; i < vector.Length; ++i)
      {
        if (vector[i] > biggestVal)
        {
          biggestVal = vector[i];
          bigIndex = i;
        }
      }
      return bigIndex;
    }
  } // NeuralNetwork
} // ns
