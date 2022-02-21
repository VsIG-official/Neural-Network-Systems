using SML.Matrices;

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

			return secondLayer;
		}

		public void Fit(double[,] xTrain, double[,] yTrain, int iterations)
		{
			for (var k = 0; k < iterations; k++)
			{
				Matrix firstMatrix = new(xTrain);

				double[,] firstLayerWeights = new double[2, 4]
				{
					{ 0.73495559, 0.08430739, 0.66952947, 0.56732125 },
					{ 0.48628304, 0.08544174, 0.49314105, 0.79496779 }
				};

				Matrix secondMatrix = new(firstLayerWeights);

				Matrix dot = firstMatrix.Multiply(secondMatrix);

				double[,] firstLayer = dot.Array;

				for (int i = 0; i < dot.Rows; i++)
				{
					for (int j = 0; j < dot.Columns; j++)
					{
						firstLayer[i, j] += bias;
					}
				}

				for (int i = 0; i < dot.Rows; i++)
				{
					for (int j = 0; j < dot.Columns; j++)
					{
						firstLayer[i, j] = Sigmoid(firstLayer[i, j]);
					}
				}

				////

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

				////

				double[,] secondLayerError = dot.Array;

				for (int i = 0; i < dot.Rows; i++)
				{
					for (int j = 0; j < dot.Columns; j++)
					{
						secondLayerError[i, j] = yTrain[i, j] - secondLayer[i, j];
					}
				}

				//////////////////
				
				Matrix secondLayerErrorMatrix = new(secondLayerError);

				for (int i = 0; i < secondLayer.GetUpperBound(0)+1; i++)
				{
					for (int j = 0; j < secondLayer.GetUpperBound(1)+1; j++)
					{
						secondLayer[i, j] = SigmoidDerivative(secondLayer[i, j]);
					}
				}

				Matrix secondLayerMatrix = new(secondLayer);

				Matrix secondLayerDeltaMatrix = secondLayerMatrix.
					Hadamard(secondLayerErrorMatrix);

				////
				
				Matrix secondLayerWeightsMatrix = new(secondLayerWeights);

				Matrix secondLayerWeightsMatrixTransposed =
					secondLayerWeightsMatrix.Transpose();

				Matrix firstLayerErrorMatrix = secondLayerDeltaMatrix.
					Multiply(secondLayerWeightsMatrixTransposed);

				////

				double[,] firstLayerDerivative = firstLayer;

				for (int i = 0; i < firstLayerDerivative.GetUpperBound(0) + 1; i++)
				{
					for (int j = 0; j < firstLayerDerivative.GetUpperBound(1) + 1; j++)
					{
						firstLayerDerivative[i, j] = SigmoidDerivative(firstLayer[i, j]);
					}
				}

				Matrix firstLayerMatrix = new(firstLayerDerivative);

				Matrix firstLayerDeltaMatrix = firstLayerMatrix.
					Hadamard(firstLayerErrorMatrix);

				////

				//double[,] secondLayerWeights1 = 
				//	{ 
				//		{ 0.45343229 },
				//		{ 0.68019613 },
				//		{ 0.10682657 },
				//		{ 0.00621122 }
				//	};

				dot = firstLayerMatrix.Transpose().Multiply(secondLayerDeltaMatrix); 

				for (int i = 0; i < secondLayerWeights.GetUpperBound(0) + 1; i++)
				{
					for (int j = 0; j < secondLayerWeights.GetUpperBound(1) + 1; j++)
					{
						secondLayerWeights[i, j] += dot[i, j];
					}
				}
			}
		}

		#endregion Methods
	}
}
