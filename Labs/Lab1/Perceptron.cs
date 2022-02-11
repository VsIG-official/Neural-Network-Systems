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
		}

		public Perceptron(double[,] input, double[] weights) : this(input)
		{
			Weights = weights;
		}

	}
}
