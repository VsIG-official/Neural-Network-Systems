﻿using SML.Matrix;

namespace Lab1
{
	public class Perceptron
	{
		#region Fields

		public double[,] Input { get; set; }
		public double[] Weights { get; set; }
		public int RunTimes { get; set; } = 1000;

		private readonly double learningRate = 0.5;
		private readonly double bias = 0.03;

		private readonly Random random = new();


		private double[,] firstLayerWeights;
		private double[,] secondLayerWeights;
		//private List<double> firstLayerWeights = new();
		//private List<double> secondLayerWeights = new();


		#endregion Fields

		#region Constructors

		public Perceptron(double[,] input)
		{
			Input = input;
			Weights = GenerateWeights();
		}

		public Perceptron(double[,] input, double[] weights) : this(input)
		{
			Weights = weights;
		}

		#endregion Constructors

		#region Methods

		private double[] GenerateWeights()
		{
			double[] weights = new double[Input.GetLength(0)];

			for (int i = 0; i < weights.Length; i++)
			{
				weights[i] = random.NextDouble();
			}

			return weights;
		}

		private double Sigmoid(double x)
		{
			return 1 / (1 + (float)Math.Exp(-x));
		}

		private double SigmoidDerivative(double x)
		{
			return Sigmoid(x) * (1 - Sigmoid(x));
		}

		public void Start()
		{
			int firstLayerLength = Input.GetUpperBound(0) + 1;
			int secondLayerLength = Input.GetUpperBound(1) + 1;

			firstLayerWeights = new double[secondLayerLength, firstLayerLength];

			for (int i = 0; i < secondLayerLength; i++)
			{
				for (int j = 0; j < firstLayerLength; j++)
				{
					firstLayerWeights[i, j] = random.NextDouble();
				}
			}

			secondLayerWeights = new double[firstLayerLength, 1];

			for (int i = 0; i < firstLayerLength; i++)
			{
				for (int j = 0; j < 1; j++)
				{
					secondLayerWeights[i, j] = random.NextDouble();
				}
			}
		}

		public double Predict(double[,] xTest)
		{
			Matrix firstMatrix = new(xTest);

			//double[,] firstLayerWeights = 
			//	{ 
			//		{ 0.96753942  , 0.85942232  , 0.93125677  , 0.54035413 },
			//		{ 0.65803162  , 0.65246242  , 0.65420492  , 0.59330152 }
			//	};

			//double[,] secondLayerWeights =
			//	{
			//		{ 0.6266687 },
			//		{ 0.69973548},
			//		{ 0.46207168 },
			//		{ 0.9354296  }
			//	};

			Matrix secondMatrix = new(firstLayerWeights);

			// Works
			Matrix dot = firstMatrix.Multiply(secondMatrix);

			double[,] firstLayer = dot.Array;

			for (int i = 0; i < dot.Rows; i++)
			{
				for (int j = 0; j < dot.Columns; j++)
				{
					firstLayer[i, j] = Sigmoid(firstLayer[i, j]);
				}
			}

			firstMatrix = new(firstLayer);

			secondMatrix = new(secondLayerWeights);

			dot = firstMatrix.Multiply(secondMatrix);

			double[,] secondLayer = dot.Array;

			for (int i = 0; i < dot.Rows; i++)
			{
				for (int j = 0; j < dot.Columns; j++)
				{
					secondLayer[i, j] = Sigmoid(secondLayer[i, j]);
				}
			}

			return 0;
		}

		#endregion Methods
	}
}
