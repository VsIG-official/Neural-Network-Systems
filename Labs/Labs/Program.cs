using Lab1;
using SML.Matrices;

namespace Labs;

internal class Program
{
    private static readonly double[,] s_input = new double[,]
    {
        { 0, 0 },
        { 0, 1 },
        { 1, 0 },
        { 1, 1 }
    };

    private static readonly double[,] s_outputs =
    {
        { 0 },
        { 1 },
        { 1 },
        { 0 }
    };

    private static readonly double[,] s_xTest = { { 1, 1 } };
    private static readonly double[,] s_xTest2 = { { 0, 0 } };
    private static readonly double[,] s_xTest3 = { { 0, 1 } };
    private static readonly double[,] s_xTest4 = { { 1, 0 } };

    private static void Main(string[] args)
    {
        Lab1();
    }

    private static void Lab1()
    {
        Perceptron perceptron = new(s_input);
        perceptron.Start();

        Console.WriteLine("Predictions before training:\n");
        RunPredictions(perceptron);

        perceptron.Train(s_input, s_outputs, 10000);

        Console.WriteLine("//////////////////////////////\n");

        Console.WriteLine("Predictions after training:\n");
        RunPredictions(perceptron);
    }

    private static void RunPredictions(Perceptron perceptron)
    {
        Matrix firstPrediction = new(perceptron.Predict(s_xTest));

        Matrix secondPrediction = new(perceptron.Predict(s_xTest2));

        Matrix thirdPrediction = new(perceptron.Predict(s_xTest3));

        Matrix fourthPrediction = new(perceptron.Predict(s_xTest4));


        Console.WriteLine("Prediction for 1, 1 is:\n");
        Console.WriteLine(firstPrediction.ToString());

        Console.WriteLine("Prediction for 0, 0 is:\n");
        Console.WriteLine(secondPrediction.ToString());

        Console.WriteLine("Prediction for 0, 1 is:\n");
        Console.WriteLine(thirdPrediction.ToString());

        Console.WriteLine("Prediction for 1, 0 is:\n");
        Console.WriteLine(fourthPrediction.ToString());
    }
}
