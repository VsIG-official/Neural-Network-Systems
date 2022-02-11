namespace Lab1
{
	public class Perceptron
	{
		public double[,] Input { get; }
		public double[] Weights { get; }
		public double[] Output { get; }

		private readonly Random random = new();

		public Perceptron(double[,] input)
		{
			Input = input;
			Output = new double[input.GetLength(0)];
			Weights = GenerateWeights();
		}

		public Perceptron(double[,] input, double[] weights) : this(input)
		{
			Weights = weights;
		}

		private double[] GenerateWeights()
		{
			double[] weights = new double[Input.GetLength(0)];

			for (int i = 0; i < weights.Length; i++)
			{
				weights[i] = random.NextDouble();
			}

			return weights;
		}
	}
}
