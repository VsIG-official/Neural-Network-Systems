using Lab1;

namespace Labs
{
    class Program
    {
        static void Main(string[] args)
        {
            double[,] input = new double[,] 
            { 
                { 1, 0 }, 
                { 1, 1 }, 
                { 0, 1 }, 
                { 0, 0 } 
            };

            double[,] outputs = 
                { 
                    { 1 }, 
                    { 0 },
                    { 1 }, 
                    { 0 } 
                };

            Perceptron perceptron = new(input);
            perceptron.Start();

            double[,] xTest = { { 1, 1 } };

            perceptron.Predict(xTest);
        }    
    }
}
