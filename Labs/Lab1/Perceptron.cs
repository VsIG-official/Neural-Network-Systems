using SML.Matrices;

namespace Lab1;

public class Perceptron
{
	#region Fields

	public double[,] Input { get; set; }
	public int RunTimes { get; set; } = 10000;

	private readonly double _bias = 0.03;

	private readonly Random _random = new();

	private double[,] _firstLayerWeights = new double[0, 0];
	private double[,] _secondLayerWeights = new double[0, 0];

	#endregion Fields

	#region Constructors

	public Perceptron(double[,] input)
	{
		Input = input;
	}

    #endregion Constructors

    #region Methods

    public void Start()
    {
        GenerateWeights();
    }

    private void GenerateWeights()
	{
		int firstLayerLength = Input.GetUpperBound(0) + 1;
		int secondLayerLength = Input.GetUpperBound(1) + 1;

		_firstLayerWeights = new double[secondLayerLength, firstLayerLength];

		for (int i = 0; i < secondLayerLength; i++)
		{
			for (int j = 0; j < firstLayerLength; j++)
			{
				_firstLayerWeights[i, j] = _random.NextDouble();
			}
		}

		_secondLayerWeights = new double[firstLayerLength, 1];

		for (int i = 0; i < firstLayerLength; i++)
		{
			for (int j = 0; j < 1; j++)
			{
				_secondLayerWeights[i, j] = _random.NextDouble();
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

	public double[,] Predict(double[,] xTest)
	{
		Matrix xTestMatrix = new(xTest);

		Matrix firstLayerWeightsMatrix = new(_firstLayerWeights);

		Matrix xTestDotfirstLayerWeights = xTestMatrix.Multiply(firstLayerWeightsMatrix);

		double[,] firstLayer = xTestDotfirstLayerWeights.Array;

		for (int i = 0; i < xTestDotfirstLayerWeights.Rows; i++)
		{
			for (int j = 0; j < xTestDotfirstLayerWeights.Columns; j++)
			{
				firstLayer[i, j] = Sigmoid(firstLayer[i, j]);
			}
		}

		Matrix firstLayerMatrix = new(firstLayer);

		Matrix secondLayerWeightsMatrix = new(_secondLayerWeights);

		Matrix firstLayerDotsecondLayerWeights = firstLayerMatrix
            .Multiply(secondLayerWeightsMatrix);

		double[,] secondLayer = firstLayerDotsecondLayerWeights.Array;

		for (int i = 0; i < firstLayerDotsecondLayerWeights.Rows; i++)
		{
			for (int j = 0; j < firstLayerDotsecondLayerWeights.Columns; j++)
			{
				secondLayer[i, j] = Sigmoid(secondLayer[i, j]);
			}
		}

		return secondLayer;
	}

	public void Train(double[,] xTrain, double[,] yTrain, int iterations)
	{
		for (var k = 0; k < iterations; k++)
		{
			Matrix xTrainMatrix = new(xTrain);

			Matrix firstLayerWeightsMatrix = new(_firstLayerWeights);

			Matrix dotXTrainAndFirstLayerWeigth = xTrainMatrix.Multiply(firstLayerWeightsMatrix);

			double[,] firstLayer = dotXTrainAndFirstLayerWeigth.Array;

			for (int i = 0; i < dotXTrainAndFirstLayerWeigth.Rows; i++)
			{
				for (int j = 0; j < dotXTrainAndFirstLayerWeigth.Columns; j++)
				{
					firstLayer[i, j] += _bias;
				}
			}

			for (int i = 0; i < dotXTrainAndFirstLayerWeigth.Rows; i++)
			{
				for (int j = 0; j < dotXTrainAndFirstLayerWeigth.Columns; j++)
				{
					firstLayer[i, j] = Sigmoid(firstLayer[i, j]);
				}
			}

			Matrix firstLayerMatrix = new(firstLayer);

			Matrix secondLayerWeightsMatrix = new(_secondLayerWeights);

			Matrix dotFirstLayerAndSecondLayerWeights =
                firstLayerMatrix.Multiply(secondLayerWeightsMatrix);

			double[,] secondLayer = dotFirstLayerAndSecondLayerWeights.Array;

			for (int i = 0; i < dotFirstLayerAndSecondLayerWeights.Rows; i++)
			{
				for (int j = 0; j < dotFirstLayerAndSecondLayerWeights.Columns; j++)
				{
					secondLayer[i, j] = Sigmoid(secondLayer[i, j]);
				}
			}

			////

			double[,] secondLayerError = dotFirstLayerAndSecondLayerWeights.Array;

			for (int i = 0; i < dotFirstLayerAndSecondLayerWeights.Rows; i++)
			{
				for (int j = 0; j < dotFirstLayerAndSecondLayerWeights.Columns; j++)
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

			Matrix dotFirstLayerAndSecondLayerDelta = firstLayerMatrix.Transpose()
                .Multiply(secondLayerDeltaMatrix); 

			for (int i = 0; i < _secondLayerWeights.GetUpperBound(0) + 1; i++)
			{
				for (int j = 0; j < _secondLayerWeights.GetUpperBound(1) + 1; j++)
				{
					_secondLayerWeights[i, j] += dotFirstLayerAndSecondLayerDelta[i, j];
				}
			}

			////

			Matrix xTrainTransposedMatrix = xTrainMatrix.Transpose();

			Matrix dotXTrainTransposedAndFirstLayerDeltaMatrix =
                xTrainTransposedMatrix.Multiply(firstLayerDeltaMatrix);

			for (int i = 0; i < _firstLayerWeights.GetUpperBound(0) + 1; i++)
			{
				for (int j = 0; j < _firstLayerWeights.GetUpperBound(1) + 1; j++)
				{
					_firstLayerWeights[i, j] += dotXTrainTransposedAndFirstLayerDeltaMatrix[i, j];
				}
			}
		}
	}

	#endregion Methods
}
