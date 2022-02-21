using Lab1;

namespace Labs
{
    class Program
    {
        static void Main(string[] args)
        {
            double[,] input = new double[,] 
            { 
                { 0, 0 }, 
                { 0, 1 }, 
                { 1, 0 }, 
                { 1, 1 } 
            };

            double[,] outputs = 
                { 
                    { 0 }, 
                    { 1 },
                    { 1 }, 
                    { 0 } 
                };

            Perceptron perceptron = new(input);
            perceptron.Start();

            double[,] xTest = { { 1, 1 } };

            perceptron.Predict(xTest);
            perceptron.Fit(input, outputs, 10000);
            double[,] pred = perceptron.Predict(xTest);
            Console.WriteLine(pred);
        }
    }
}
