using Microsoft.ML;
using Microsoft.ML.Core.Data;
using System.IO;
using System.Net.Http;

namespace TaxiFare.Service
{
    public class HttpLoader : IModelLoader
    {
        private readonly string modelUri;
        private readonly Stream stream;
        public HttpLoader(Stream stream, string uri)
        {
            this.modelUri = uri;
            this.stream = stream;
        }
        public ITransformer Load(MLContext context)
        {
            ITransformer LoadedModel;

            LoadedModel = context.Model.Load(stream);
            return LoadedModel;
        }

    }
}