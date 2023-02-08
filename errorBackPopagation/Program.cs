namespace errorBackPopagation
{
    class NeuralNetwork
    {
        private List<Neuron> _hiddenLayer;
        private Neuron _outNeuron;
        private double[] _input;
        private double[] _outHidden;
        private double _output;

        public NeuralNetwork(int countOfHiddenNeurons)
        {
            _input = new double[2];
            _hiddenLayer = new List<Neuron>();
            _outHidden = new double[countOfHiddenNeurons];

            for (int i = 0; i < countOfHiddenNeurons; i++)
            {
                _hiddenLayer.Add(new Neuron(2));
            }

            _outNeuron = new Neuron(countOfHiddenNeurons);
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
                    this._input = input[i];

                    for (int j = 0; j < _hiddenLayer.Count; j++)
                    {
                        _hiddenLayer[j].ForwardMovement(this._input);
                        _outHidden[j] = _hiddenLayer[j].Y;
                    }

                    _outNeuron.ForwardMovement(_outHidden);
                    _output = _outNeuron.Y;

                    _outNeuron.ErrorBackForLastNeuron(D[i]);
                    for (int j = _hiddenLayer.Count - 1; j >= 0; j--)
                    {
                        _hiddenLayer[j].ErrorBack(_outNeuron.W[j], _outNeuron.E);
                    }

                    _outNeuron.ChangeWeigth();
                    for (int j = _hiddenLayer.Count - 1; j >= 0; j--)
                    {
                        _hiddenLayer[j].ChangeWeigth();
                    }
                }
            } while (iterations < 500);

            Console.WriteLine("Обучение завершено!\n");
        }

        public void SolutionOfProblem(double[] input)
        {
            this._input = input;

            for (int i = 0; i < _hiddenLayer.Count; i++)
            {
                _hiddenLayer[i].ForwardMovement(input);
                _outHidden[i] = _hiddenLayer[i].Y;
            }

            _outNeuron.ForwardMovement(_outHidden);
            _output = _outNeuron.Y;


            Console.WriteLine($"X1 = {this._input[0]} X2 = {this._input[1]} Y = {_output}");
        }
    }

    class Neuron
    {
        public double[] W { get; set; }
        public double E { get; set; }
        public double Y { get; set; }

        private const int _x0 = 1;
        private double[] _x;
        private double _b = 1;
        private double _netRes;
        private double _n = 8;

        public Neuron(int countOfInputs)
        {
            Random rnd = new Random();
            W = new double[countOfInputs];
            _x = new double[countOfInputs];
            for (int i = 0; i < countOfInputs; i++)
            {
                W[i] = rnd.NextDouble() + 0.2;
            }
        }

        public void Net()
        {
            _netRes = 0;

            for (int i = 0; i < _x.Length; i++)
            {
                _netRes += W[i] * _x[i];
            }

            _netRes += _b * _x0;
        }

        public void ActivateFunction()
        {
            Y = 1 / (1 + Math.Exp(-_netRes));
        }

        public void ForwardMovement(double[] X)
        {
            for (int i = 0; i < X.Length; i++)
            {
                this._x[i] = X[i];
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
            for (int i = 0; i < _x.Length; i++)
            {
                W[i] -= _n * E * _x[i];
            }
            _b -= _n * E * _x0;
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

            NeuralNetwork networkOne = new NeuralNetwork(2);

            networkOne.LearningNeuralNetwork(input, D);

            Console.WriteLine("Решение задачи:");

            for (int i = input.Length - 1; i >= 0; i--)
            {
                networkOne.SolutionOfProblem(input[i]);
            }
        }
    }
}