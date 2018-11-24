using Microsoft.ML;
using Microsoft.ML.Core.Data;
using System.IO;

namespace TaxiFare.Service
{
    public class FileLoader : IModelLoader
    {
        private readonly string modelPath;
        public FileLoader(string modelPath)
        {
            this.modelPath = modelPath;
        }
        public ITransformer Load(MLContext context)
        {
            ITransformer LoadedModel;
            using (var stream = new FileStream(modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                LoadedModel = context.Model.Load(stream);
            return LoadedModel;
        }

    }
}