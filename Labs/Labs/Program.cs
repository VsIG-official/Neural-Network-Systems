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

            int[] outputs = { 1, 0, 1, 0 };

            Perceptron perceptron = new(input);
        }    
    }
}