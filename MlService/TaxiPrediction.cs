using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Runtime.Data;
using System;
using System.IO;
using TaxiFare.Shared;

namespace TaxiFare.Service
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

}