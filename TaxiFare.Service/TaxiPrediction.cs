using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;
using System.Threading.Tasks;
using TaxiFare.Shared;

namespace TaxiFare.Service
{
    public class TaxiPrediction : ITaxiPrediction
    {
        private readonly ITransformer model;
        private readonly MLContext context;
        public TaxiPrediction(IModelLoader loader)
        {
            context = new MLContext(seed: 0);
            model = loader.Load(context);

        }
        public TaxiTripFarePrediction GetTripFare(TaxiTrip taxiTrip)
        {
            var predictionFunction = model.MakePredictionFunction<TaxiTrip, TaxiTripFarePrediction>(context);

            return predictionFunction.Predict(taxiTrip);
        }

    }

    public interface IModelLoader
    {
        ITransformer Load(MLContext context);
    }
}