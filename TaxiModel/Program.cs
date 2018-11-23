using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms.Categorical;
using System;
using System.IO;
using TaxiFare.Shared;

namespace TaxiFare.TrainingModel
{
    class Program
    {

        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static readonly TaxiTrip SampleData = new TaxiTrip()
        {
            VendorId = "VTS",
            RateCode = "1",
            PassengerCount = 1,
            TripTime = 1140,
            TripDistance = 3.75f,
            PaymentType = "CRD",
            FareAmount = 0 // To predict. Actual/Observed = 15.5
        };

        static void Main(string[] args)
        {

            MLContext mlContext = new MLContext(seed: 0);

            // Train
            Console.WriteLine("Create and Train the Model...");
            var model = Train(mlContext);
            Console.WriteLine("End of training.");
            SaveModelAsFile(mlContext, model);
            Console.WriteLine("The model is saved to {0}", _modelPath);

            // Evaluate
            var metrics = Evaluate(mlContext, model);
            DisplayMetrics(metrics);

            // Test
            var prediction = TestSinglePrediction(mlContext, SampleData);
            DisplayPredictedFare(prediction);

            Console.ReadKey();
        }
        static IDataView TrainDataReader(MLContext mlContext) => CreateTextLoader(mlContext)
                .Read(Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv"));

        public static ITransformer Train(MLContext mlContext) => mlContext.Transforms.CopyColumns("FareAmount", "Label")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentType"))
                .Append(mlContext.Transforms.Concatenate("Features", "VendorId", "RateCode", "PassengerCount", "TripTime", "TripDistance", "PaymentType"))
                .Append(mlContext.Regression.Trainers.FastTree())
                .Fit(TrainDataReader(mlContext));

        static IDataView TestDataReader(MLContext mlContext) => CreateTextLoader(mlContext)
                     .Read(Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv"));

        private static RegressionEvaluator.Result Evaluate(MLContext mlContext, ITransformer model)
        {
            var predictions = model.Transform(TestDataReader(mlContext));
            return mlContext.Regression.Evaluate(predictions, "Label", "Score");
        }

        private static void DisplayMetrics(RegressionEvaluator.Result metrics)
        {
            Console.WriteLine();
            Console.WriteLine(new String('=', 35));
            Console.WriteLine("Model quality metrics evaluation");
            Console.WriteLine();
            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            Console.WriteLine($"*       RMS loss:      {metrics.Rms:#.##}");
            Console.WriteLine(new String('=', 35));
        }

        private static TaxiTripFarePrediction TestSinglePrediction(MLContext mlContext, TaxiTrip sample)
        {
            ITransformer loadedModel;

            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                loadedModel = mlContext.Model.Load(stream);

            var predictionFunction = loadedModel.MakePredictionFunction<TaxiTrip, TaxiTripFarePrediction>(mlContext);

            return predictionFunction.Predict(sample);
        }

        private static void DisplayPredictedFare(TaxiTripFarePrediction prediction)
        {
            Console.WriteLine();
            Console.WriteLine(new String('*', 35));
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fileStream);
        }

        static TextLoader CreateTextLoader(MLContext mlContext) => mlContext.Data.TextReader(
            new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("VendorId", DataKind.Text, 0),
                    new TextLoader.Column("RateCode", DataKind.Text, 1),
                    new TextLoader.Column("PassengerCount", DataKind.R4, 2),
                    new TextLoader.Column("TripTime", DataKind.R4, 3),
                    new TextLoader.Column("TripDistance", DataKind.R4, 4),
                    new TextLoader.Column("PaymentType", DataKind.Text, 5),
                    new TextLoader.Column("FareAmount", DataKind.R4, 6)
                }
            }
        );
    }
}
