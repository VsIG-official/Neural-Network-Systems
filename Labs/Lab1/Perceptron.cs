using SML.Matrix;

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
			double first_layer_length = Input.GetUpperBound(0) + 1;
			double second_layer_length = Input.GetUpperBound(1) + 1;

			List<double> first_layer_weights = new();

			for (int i = 0; i < first_layer_length * 2; i++)
			{
				first_layer_weights.Add(random.NextDouble());
			}

			List<double> second_layer_weights = new();

			for (int i = 0; i < second_layer_length * 2; i++)
			{
				second_layer_weights.Add(random.NextDouble());
			}

			double[,] first = { { 1, 1 } };
			double[,] second = { { 0.62013305, 0.79208935, 0.02272945, 0.85735877 },
				{ 0.91331244, 0.78375064, 0.04980426, 0.30849369 } };

			Matrix firstMatrix = new(first);
			Matrix secondMatrix = new(second);

			Matrix dot = firstMatrix.Multiply(secondMatrix);
		}

		#endregion Methods
	}
}
