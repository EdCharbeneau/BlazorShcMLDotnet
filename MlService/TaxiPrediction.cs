using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using System.IO;
using TaxiFare.Shared;

namespace TaxiFare.Service
{
    public class TaxiPrediction
    {
        protected readonly ITransformer loadedModel;
        protected readonly MLContext mlContext;
        public TaxiPrediction(string modelPath)
        {
            mlContext = new MLContext(seed: 0);

            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                loadedModel = mlContext.Model.Load(stream);
        }

        public TaxiTripFarePrediction GetTripFare(TaxiTrip taxiTrip)
        {
            var predictionFunction = loadedModel.MakePredictionFunction<TaxiTrip, TaxiTripFarePrediction>(mlContext);

            return predictionFunction.Predict(taxiTrip);
        }
    }

}