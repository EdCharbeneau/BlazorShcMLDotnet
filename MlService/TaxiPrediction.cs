using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.IO;

namespace MlService
{
    public class TaxiPrediction
    {

        public static TaxiTripFarePrediction GetTripFare(TaxiTrip taxiTrip)
        {
            string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "TrainedTaxiModel.zip");

            MLContext mlContext = new MLContext(seed: 0);

            ITransformer loadedModel;
            // Loading TrainedTaxiModel.zip
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }
            // Func<TaxiTrip, TaxiTripFarePrediction>
            var predictionFunction = loadedModel.MakePredictionFunction<TaxiTrip, TaxiTripFarePrediction>(mlContext);

            return predictionFunction.Predict(taxiTrip);

        }
    }

    public class TaxiTrip
    {
        [Column("0")]
        public string VendorId;

        [Column("1")]
        public string RateCode;

        [Column("2")]
        public float PassengerCount;

        [Column("3")]
        public float TripTime;

        [Column("4")]
        public float TripDistance;

        [Column("5")]
        public string PaymentType;

        [Column("6")]
        public float FareAmount;
    }
    public class TaxiTripFarePrediction
    {
        [ColumnName("Score")]
        public float FareAmount;
    }
}