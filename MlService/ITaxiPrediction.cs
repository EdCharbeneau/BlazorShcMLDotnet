using TaxiFare.Shared;

namespace TaxiFare.Service
{
    public interface ITaxiPrediction
    {
        TaxiTripFarePrediction GetTripFare(TaxiTrip taxiTrip);
    }
}