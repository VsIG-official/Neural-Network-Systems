using SML.Matrices;

namespace Lab1
{
	public class Perceptron
	{
		#region Fields

		public double[,] Input { get; set; }
		public double[] Weights { get; set; }
		public int RunTimes { get; set; } = 1000;

		private readonly double bias = 0.03;

		private readonly Random random = new();

		private double[,] firstLayerWeights = new double[0, 0];
		private double[,] secondLayerWeights = new double[0, 0];

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

		public double Sigmoid(double x)
		{
			return 1 / (1 + (float)Math.Exp(-x));
		}

		public double SigmoidDerivative(double x)
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

		public double[,] Predict(double[,] xTest)
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

			Matrix dot = firstMatrix.Multiply(secondMatrix);

			double[,] firstLayer = dot.Array;

			for (int i = 0; i < dot.Rows; i++)
			{
				for (int j = 0; j < dot.Columns; j++)
				{
					firstLayer[i, j] = Sigmoid(firstLayer[i, j]);
				}
			}

			Matrix firstMatrix1 = new(firstLayer);

			Matrix secondMatrix1 = new(secondLayerWeights);

			Matrix dot1 = firstMatrix1.Multiply(secondMatrix1);

			double[,] secondLayer = dot1.Array;

			for (int i = 0; i < dot1.Rows; i++)
			{
				for (int j = 0; j < dot1.Columns; j++)
				{
					secondLayer[i, j] = Sigmoid(secondLayer[i, j]);
				}
			}

			return secondLayer;
		}

		public void Fit(double[,] xTrain, double[,] yTrain, int iterations)
		{
			for (var k = 0; k < iterations; k++)
			{
				Matrix xTrainMatrix = new(xTrain);

					//Console.WriteLine(xTrainMatrix.ToString());

				double[,] firstLayerWeights = new double[2, 4]
				{
					{ 0.81377233, 0.36367063, 0.9062002, 0.13996215 },
					{ 0.95584121, 0.82865029, 0.50139206, 0.50926942 }
				};

				Matrix firstLayerWeightsMatrix = new(firstLayerWeights);

					//Console.WriteLine(firstLayerWeightsMatrix.ToString());

				Matrix xTrainDotFirstLayerWeigth = xTrainMatrix.Multiply(firstLayerWeightsMatrix);

					//Console.WriteLine(xTrainDotFirstLayerWeigth.ToString());

				double[,] firstLayer = xTrainDotFirstLayerWeigth.Array;

				for (int i = 0; i < xTrainDotFirstLayerWeigth.Rows; i++)
				{
					for (int j = 0; j < xTrainDotFirstLayerWeigth.Columns; j++)
					{
						firstLayer[i, j] += bias;
					}
				}

				for (int i = 0; i < xTrainDotFirstLayerWeigth.Rows; i++)
				{
					for (int j = 0; j < xTrainDotFirstLayerWeigth.Columns; j++)
					{
						firstLayer[i, j] = Sigmoid(firstLayer[i, j]);
					}
				}

				////

				Matrix firstLayerMatrix1 = new(firstLayer);

					//Console.WriteLine(firstLayerMatrix1.ToString());

				double[,] secondLayerWeights = new double[4, 1]
				{
					{ 0.16980977 },
					{ 0.92468996 },
					{ 0.27498607 },
					{ 0.07080158 }
				};

				Matrix secondLayerWeightsMatrix1 = new(secondLayerWeights);

					//Console.WriteLine(secondLayerWeightsMatrix1.ToString());

				Matrix dot1 = firstLayerMatrix1.Multiply(secondLayerWeightsMatrix1);

					//Console.WriteLine(dot1.ToString());

				double[,] secondLayer = dot1.Array;

				for (int i = 0; i < dot1.Rows; i++)
				{
					for (int j = 0; j < dot1.Columns; j++)
					{
						secondLayer[i, j] = Sigmoid(secondLayer[i, j]);
					}
				}

				Matrix secondLayerMatrix = new(secondLayer);

					//Console.WriteLine(secondLayerMatrix.ToString());

				////

				double[,] secondLayerError = dot1.Array;

				for (int i = 0; i < dot1.Rows; i++)
				{
					for (int j = 0; j < dot1.Columns; j++)
					{
						secondLayerError[i, j] = yTrain[i, j] - secondLayer[i, j];
					}
				}

				//////////////////
				
				Matrix yTrainMatrix = new(yTrain);

					//Console.WriteLine(yTrainMatrix.ToString());

				Matrix secondLayerErrorMatrix = new(secondLayerError);

					//Console.WriteLine(secondLayerErrorMatrix.ToString());

				for (int i = 0; i < secondLayer.GetUpperBound(0)+1; i++)
				{
					for (int j = 0; j < secondLayer.GetUpperBound(1)+1; j++)
					{
						secondLayer[i, j] = SigmoidDerivative(secondLayer[i, j]);
					}
				}

				Matrix secondLayerMatrix1 = new(secondLayer);

					//Console.WriteLine(secondLayerMatrix1.ToString());

				Matrix secondLayerDeltaMatrix = secondLayerMatrix1.
					Hadamard(secondLayerErrorMatrix);

					//Console.WriteLine(secondLayerDeltaMatrix.ToString());

				////

				Matrix secondLayerWeightsMatrix = new(secondLayerWeights);

				Matrix secondLayerWeightsMatrixTransposed =
					secondLayerWeightsMatrix.Transpose();

					//Console.WriteLine(secondLayerWeightsMatrixTransposed.ToString());

				Matrix firstLayerErrorMatrix = secondLayerDeltaMatrix.
					Multiply(secondLayerWeightsMatrixTransposed);

					//Console.WriteLine(firstLayerErrorMatrix.ToString());

				////

				double[,] firstLayerDerivative = firstLayer;

				for (int i = 0; i < firstLayerDerivative.GetUpperBound(0) + 1; i++)
				{
					for (int j = 0; j < firstLayerDerivative.GetUpperBound(1) + 1; j++)
					{
						firstLayerDerivative[i, j] = SigmoidDerivative(firstLayer[i, j]);
					}
				}

				Matrix firstLayerDerivativeMatrix = new(firstLayerDerivative);

					//Console.WriteLine(firstLayerDerivativeMatrix.ToString());

				Matrix firstLayerDeltaMatrix = firstLayerDerivativeMatrix.
					Hadamard(firstLayerErrorMatrix);

					//Console.WriteLine(firstLayerDeltaMatrix.ToString());

				////

				//double[,] secondLayerWeights1 = 
				//	{ 
				//		{ 0.45343229 },
				//		{ 0.68019613 },
				//		{ 0.10682657 },
				//		{ 0.00621122 }
				//	};

					//Console.WriteLine(firstLayerMatrix1.Transpose().ToString());

					//Console.WriteLine(secondLayerDeltaMatrix.ToString());

				Matrix dot2 = firstLayerMatrix1.Transpose().Multiply(secondLayerDeltaMatrix); 

				for (int i = 0; i < secondLayerWeights.GetUpperBound(0) + 1; i++)
				{
					for (int j = 0; j < secondLayerWeights.GetUpperBound(1) + 1; j++)
					{
						secondLayerWeights[i, j] += dot2[i, j];
					}
				}

					//Console.WriteLine(dot2.ToString());

				Matrix secondLayerWeightsMatrix2 = new(secondLayerWeights);

					//Console.WriteLine(secondLayerWeightsMatrix2.ToString());

				////

				Matrix firstMatrix2 = new(xTrain);
				firstMatrix2 = firstMatrix2.Transpose();

					//Console.WriteLine(firstMatrix2.ToString());

					//Console.WriteLine(firstLayerDeltaMatrix.ToString());

				Matrix dot3 = firstMatrix2.Multiply(firstLayerDeltaMatrix);

					//Console.WriteLine(dot3.ToString());

				for (int i = 0; i < firstLayerWeights.GetUpperBound(0) + 1; i++)
				{
					for (int j = 0; j < firstLayerWeights.GetUpperBound(1) + 1; j++)
					{
						firstLayerWeights[i, j] += dot3[i, j];
					}
				}

				Matrix firstLayerWeightsMatrix1 = new(firstLayerWeights);

					//Console.WriteLine(firstLayerWeightsMatrix1.ToString());
			}
		}

		#endregion Methods
	}
}
