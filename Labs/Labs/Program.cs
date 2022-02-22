using Lab1;
using SML.Matrices;

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

            double[,] xTest = { { 0, 1 } };
            double[,] xTest2 = { { 1, 1 } };
            double[,] xTest3 = { { 0, 0 } };
            double[,] xTest4 = { { 1, 0 } };

            perceptron.Predict(input);

            perceptron.Fit(input, outputs, 10000);

            perceptron.Predict(input);

            Matrix firstPrediction = new(perceptron.Predict(xTest));

                Console.WriteLine(firstPrediction.ToString());

            Matrix secondPrediction = new(perceptron.Predict(xTest2));

                Console.WriteLine(secondPrediction.ToString());

            Matrix thirdPrediction = new(perceptron.Predict(xTest3));

                Console.WriteLine(thirdPrediction.ToString());

            Matrix fourthPrediction = new(perceptron.Predict(xTest4));

                Console.WriteLine(fourthPrediction.ToString());
        }
    }
}
