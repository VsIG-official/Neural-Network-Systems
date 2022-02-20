using System.Numerics;

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
			double first_layer_length = Input.GetUpperBound(0);
			double second_layer_length = Input.GetUpperBound(1);

			double first_layer_weights = random.NextDouble() * 
				(first_layer_length - second_layer_length) + second_layer_length;

			double second_layer_weights = random.NextDouble() *
				(1 - first_layer_length) + first_layer_length;
		}

		#endregion Methods
	}
}
