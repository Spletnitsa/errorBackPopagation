namespace errorBackPopagation
{
    class neuralNetwork
    {
        private List<neuron> hiddenLayer;
        private neuron outNeuron;
        private double[] input;
        private double[] outHidden;
        private double output;

        public neuralNetwork(int countOfHiddenNeurons)
        {
            input = new double[2];
            hiddenLayer = new List<neuron>();
            outHidden = new double[countOfHiddenNeurons];

            for (int i = 0; i < countOfHiddenNeurons; i++)
            {
                hiddenLayer.Add(new neuron(2));
            }

            outNeuron = new neuron(4);
        }

        public void LearningNeuralNetwork(double[][] input, int[] D)
        {
            int iterations = 0;

            Console.WriteLine("Начато обучение\n");
            do
            {
                iterations++;

                for (int i = 0; i < input.Length; i++)
                {
                    this.input = input[i];

                    for (int j = 0; j < hiddenLayer.Count; j++)
                    {
                        hiddenLayer[j].ForwardMovement(this.input);
                        outHidden[j] = hiddenLayer[j].Y;
                    }

                    outNeuron.ForwardMovement(outHidden);
                    output = outNeuron.Y;

                    outNeuron.ErrorBackForLastNeuron(D[i]);
                    for (int j = hiddenLayer.Count - 1; j >= 0; j--)
                    {
                        hiddenLayer[j].ErrorBack(outNeuron.W[j], outNeuron.E);
                    }

                    outNeuron.ChangeWeigth();
                    for (int j = hiddenLayer.Count - 1; j >= 0; j--)
                    {
                        hiddenLayer[j].ChangeWeigth();
                    }
                }
            } while (iterations < 10000);

            Console.WriteLine("Обучение завершено!\n");
        }

        public void SolutionOfProblem(double[] input)
        {
            this.input = input;

            for (int i = 0; i < hiddenLayer.Count; i++)
            {
                hiddenLayer[i].ForwardMovement(input);
                outHidden[i] = hiddenLayer[i].Y;
            }

            outNeuron.ForwardMovement(outHidden);
            output = outNeuron.Y;


            Console.WriteLine($"X1 = {this.input[0]} X2 = {this.input[1]} Y = {output}");
        }
    }

    class neuron
    {
        public double[] W { get; set; }
        public double E { get; set; }
        public double Y { get; set; }

        private double[] X;
        private const int X0 = 1;
        private double b = 1;
        private double NetRes;
        private double N = 0.5;

        public neuron(int countOfInputs)
        {
            Random rnd = new Random();
            W = new double[countOfInputs];
            X = new double[countOfInputs];
            for (int i = 0; i < countOfInputs; i++)
            {
                W[i] = rnd.NextDouble() + 0.2;
            }
        }

        public void Net()
        {
            NetRes = 0;

            for (int i = 0; i < X.Length; i++)
            {
                NetRes += W[i] * X[i];
            }

            NetRes += b * X0;
        }

        public void ActivateFunction()
        {
            Y = 1 / (1 + Math.Exp(-NetRes));
        }

        public void ForwardMovement(double[] X)
        {
            for (int i = 0; i < X.Length; i++)
            {
                this.X[i] = X[i];
            }

            Net();
            ActivateFunction();
        }

        public void ErrorBackForLastNeuron(double D)
        {
            E = -(D - Y) * Y * (1 - Y);
        }

        public void ErrorBack(double outputWeight, double previousError)
        {
            E = Y * (1 - Y) * outputWeight * previousError;
        }

        public void ChangeWeigth()
        {
            for (int i = 0; i < X.Length; i++)
            {
                W[i] -= N * E * X[i];
            }
            b -= N * E * X0;
        }
    }
    internal class Program
    {
        static void Main(string[] args)
        {
            double[][] input = new double[4][];
            input[0] = new double[] { 0, 0 };
            input[1] = new double[] { 0, 1 };
            input[2] = new double[] { 1, 0 };
            input[3] = new double[] { 1, 1 };
            int[] D = new int[] {0, 1, 1, 0};

            neuralNetwork networkOne = new neuralNetwork(4);

            networkOne.LearningNeuralNetwork(input, D);

            Console.WriteLine("Решение задачи:");

            for (int i = 0; i < input.Length; i++)
            {
                networkOne.SolutionOfProblem(input[i]);
            }
        }
    }
}