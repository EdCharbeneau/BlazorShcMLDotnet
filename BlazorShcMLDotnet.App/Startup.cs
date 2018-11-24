using Microsoft.AspNetCore.Blazor.Builder;
using Microsoft.Extensions.DependencyInjection;
using BlazorShcMLDotnet.App.Services;
using TaxiFare.Service;
using System.IO;
using System;

namespace BlazorShcMLDotnet.App
{
    public class Startup
    {
        public void ConfigureServices(IServiceCollection services)
        {
            // Since Blazor is running on the server, we can use an application service
            // to read the forecast data.
            services.AddSingleton<WeatherForecastService>();
            var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "TrainedTaxiModel.zip");
            var loader = new FileLoader(modelPath);
            services.AddSingleton<ITaxiPrediction>(sp => new TaxiPrediction(loader));
        }

        public void Configure(IBlazorApplicationBuilder app)
        {
            app.AddComponent<App>("app");
        }
    }
}
