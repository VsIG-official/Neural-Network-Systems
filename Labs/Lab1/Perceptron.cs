using SML.Matrices;

namespace Lab1
{
	public class Perceptron
	{
		#region Fields

		public double[,] Input { get; set; }
		public int RunTimes { get; set; } = 10000;

		private readonly double bias = 0.03;

		private readonly Random random = new();

		private double[,] firstLayerWeights = new double[0, 0];
		private double[,] secondLayerWeights = new double[0, 0];

		#endregion Fields

		#region Constructors

		public Perceptron(double[,] input)
		{
			Input = input;
		}

		#endregion Constructors

		#region Methods

		private void GenerateWeights()
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

		private static double Sigmoid(double x)
		{
			return 1 / (1 + (float)Math.Exp(-x));
		}

		private static double SigmoidDerivative(double x)
		{
			return Sigmoid(x) * (1 - Sigmoid(x));
		}

		public void Start()
		{
			GenerateWeights();
		}

		public double[,] Predict(double[,] xTest)
		{
			Matrix firstMatrix = new(xTest);

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

				Matrix firstLayerWeightsMatrix = new(firstLayerWeights);

				Matrix xTrainDotFirstLayerWeigth = xTrainMatrix.Multiply(firstLayerWeightsMatrix);

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

				Matrix secondLayerWeightsMatrix1 = new(secondLayerWeights);

				Matrix dot1 = firstLayerMatrix1.Multiply(secondLayerWeightsMatrix1);

				double[,] secondLayer = dot1.Array;

				for (int i = 0; i < dot1.Rows; i++)
				{
					for (int j = 0; j < dot1.Columns; j++)
					{
						secondLayer[i, j] = Sigmoid(secondLayer[i, j]);
					}
				}

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

				Matrix secondLayerErrorMatrix = new(secondLayerError);

				for (int i = 0; i < secondLayer.GetUpperBound(0)+1; i++)
				{
					for (int j = 0; j < secondLayer.GetUpperBound(1)+1; j++)
					{
						secondLayer[i, j] = SigmoidDerivative(secondLayer[i, j]);
					}
				}

				Matrix secondLayerMatrix1 = new(secondLayer);

				Matrix secondLayerDeltaMatrix = secondLayerMatrix1.
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

				Matrix firstLayerDerivativeMatrix = new(firstLayerDerivative);

				Matrix firstLayerDeltaMatrix = firstLayerDerivativeMatrix.
					Hadamard(firstLayerErrorMatrix);

				////

				Matrix dot2 = firstLayerMatrix1.Transpose().Multiply(secondLayerDeltaMatrix); 

				for (int i = 0; i < secondLayerWeights.GetUpperBound(0) + 1; i++)
				{
					for (int j = 0; j < secondLayerWeights.GetUpperBound(1) + 1; j++)
					{
						secondLayerWeights[i, j] += dot2[i, j];
					}
				}

				////

				Matrix firstMatrix2 = new(xTrain);
				firstMatrix2 = firstMatrix2.Transpose();

				Matrix dot3 = firstMatrix2.Multiply(firstLayerDeltaMatrix);


				for (int i = 0; i < firstLayerWeights.GetUpperBound(0) + 1; i++)
				{
					for (int j = 0; j < firstLayerWeights.GetUpperBound(1) + 1; j++)
					{
						firstLayerWeights[i, j] += dot3[i, j];
					}
				}
			}
		}

		#endregion Methods
	}
}
