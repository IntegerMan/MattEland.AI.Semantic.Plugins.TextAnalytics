using Azure;
using Azure.AI.TextAnalytics;
using Microsoft.SemanticKernel;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MattEland.AI.Semantic.Plugins.TextAnalytics
{
    public class TextAnalyticsPlugin
    {
        private readonly TextAnalyticsClient _textClient;

        public TextAnalyticsPlugin(string endpoint, string key) : this(new Uri(endpoint), key)
        {
        }

        public TextAnalyticsPlugin(Uri endpoint, string key) : this(endpoint, new AzureKeyCredential(key))
        {
        }

        public TextAnalyticsPlugin(Uri endpoint, AzureKeyCredential credential)
        {
            _textClient = new TextAnalyticsClient(endpoint, credential);
        }

        public TextAnalyticsPlugin(TextAnalyticsClient client)
        {
            _textClient = client;
        }

        [KernelFunction, Description("Given some text, return an analysis of the overall sentiment and how likely it was the sentiment was positive, negative, and neutral")]
        public async Task<string> AnalyzeSentiment([Description("The text to analyze")] string text)
        {
            try
            {
                Response<DocumentSentiment> response = await _textClient.AnalyzeSentimentAsync(text);

                DocumentSentiment documentSentiment = response.Value;

                StringBuilder sb = new StringBuilder();

                sb.Append($"The text has an overall sentiment of {documentSentiment.Sentiment}.");
                sb.Append($"It is {documentSentiment.ConfidenceScores.Positive:P0} likely the sentiment was positive, ");
                sb.Append($"{documentSentiment.ConfidenceScores.Negative:P0} likely the sentiment was negative, ");
                sb.Append($"and {documentSentiment.ConfidenceScores.Neutral:P0} likely the sentiment was neutral.");

                return documentSentiment.Sentiment.ToString();
            }
            catch (RequestFailedException ex)
            {
                return $"Could not analyze sentiment: {ex.Message}";
            }
        }

        [KernelFunction, Description("Given some text, return a summarization of the text")]
        public async Task<string> SummarizeText([Description("The text to analyze")] string text)
        {
            try
            {
                TextAnalyticsActions actions = new TextAnalyticsActions()
                {
                    AbstractiveSummarizeActions = new List<AbstractiveSummarizeAction>() { new AbstractiveSummarizeAction() },
                };

                string[] documents = new string[] { text };
                AnalyzeActionsOperation operation = await _textClient.AnalyzeActionsAsync(WaitUntil.Completed, documents, actions);

                StringBuilder sb = new StringBuilder();
                await foreach (AnalyzeActionsResult result in operation.Value)
                {
                    foreach (AbstractiveSummarizeResult abstractResult in result.AbstractiveSummarizeResults.SelectMany(r => r.DocumentsResults))
                    {
                        if (abstractResult.HasError)
                        {
                            sb.AppendLine($"Summarization failed: {abstractResult.Error.ErrorCode} - {abstractResult.Error.Message}");
                            continue;
                        }

                        foreach (AbstractiveSummary summary in abstractResult.Summaries)
                        {
                            sb.AppendLine(summary.Text);
                        }
                    }
                }

                return sb.ToString();
            }
            catch (RequestFailedException ex)
            {
                return GetRequestFailedString(ex);
            }
        }

        [KernelFunction, Description("Given some text, identify the entities mentioned in the text and provide additional links to each if able")]
        public async Task<string> IdentifyEntities([Description("The text to analyze")] string text)
        {
            try
            {
                TextAnalyticsActions actions = new TextAnalyticsActions()
                {
                    RecognizeEntitiesActions = new List<RecognizeEntitiesAction>() { new RecognizeEntitiesAction() },
                    RecognizeLinkedEntitiesActions = new List<RecognizeLinkedEntitiesAction>() { new RecognizeLinkedEntitiesAction() },
                };

                string[] documents = new string[] { text };
                AnalyzeActionsOperation operation = await _textClient.AnalyzeActionsAsync(WaitUntil.Completed, documents, actions);

                Dictionary<string, string> entities = new Dictionary<string, string>();

                await foreach (AnalyzeActionsResult result in operation.Value)
                {
                    // Start with standard entity results
                    foreach (RecognizeEntitiesResult entityResult in result.RecognizeEntitiesResults.SelectMany(r => r.DocumentsResults))
                    {
                        foreach (CategorizedEntity entity in entityResult.Entities)
                        {
                            entities[entity.Text] = entity.Category.ToString();
                        }
                    }

                    // Add linked entities to the results, replacing any existing entities with the same name if needed
                    foreach (RecognizeLinkedEntitiesResult linkedEntityResult in result.RecognizeLinkedEntitiesResults.SelectMany(r => r.DocumentsResults))
                    {
                        foreach (LinkedEntity entity in linkedEntityResult.Entities)
                        {
                            entities[entity.Name] = entity.Url.ToString();
                        }
                    }
                }

                if (entities.Count == 0)
                {
                    return "No entities were found in the text.";
                }

                StringBuilder sb = new StringBuilder();
                sb.AppendLine("Entities found:");
                foreach (KeyValuePair<string, string> entity in entities)
                {
                    sb.AppendLine($"{entity.Key}: {entity.Value}");
                }
                return sb.ToString();
            }
            catch (RequestFailedException ex)
            {
                return GetRequestFailedString(ex);
            }
        }

        [KernelFunction, Description("Given some text, identify potential sensitive strings in the text")]
        public async Task<string> IdentifySensitiveInformation([Description("The text to analyze")] string text)
        {
            try
            {
                TextAnalyticsActions actions = new TextAnalyticsActions()
                {
                    RecognizePiiEntitiesActions = new List<RecognizePiiEntitiesAction>() { new RecognizePiiEntitiesAction() },
                };

                string[] documents = new string[] { text };
                AnalyzeActionsOperation operation = await _textClient.AnalyzeActionsAsync(WaitUntil.Completed, documents, actions);

                List<PiiEntity> entries = new List<PiiEntity>();
                await foreach (AnalyzeActionsResult result in operation.Value)
                {
                    foreach (RecognizePiiEntitiesResult piiEntityResult in result.RecognizePiiEntitiesResults.SelectMany(r => r.DocumentsResults))
                    {
                        if (piiEntityResult.HasError)
                        {
                            return $"Pii entity recognition failed: {piiEntityResult.Error.ErrorCode} - {piiEntityResult.Error.Message}";
                        }

                        foreach (PiiEntity entity in piiEntityResult.Entities)
                        {
                            entries.Add(entity);
                        }
                    }
                }

                if (entries.Count == 0)
                {
                    return "No sensitive information was found.";
                }

                StringBuilder sb = new StringBuilder();
                sb.AppendLine("Sensitive information found:");
                foreach (PiiEntity entity in entries)
                {
                    sb.AppendLine($"- {entity.Text} ({entity.Category})");
                }
                return sb.ToString();
            }
            catch (RequestFailedException ex)
            {
                return GetRequestFailedString(ex);
            }
        }


        [KernelFunction, Description("Given some text, return a summarization of the text with accompanying sentence extracts")]
        public async Task<string> SummarizeTextWithExtracts([Description("The text to analyze")] string text)
        {
            try
            {
                TextAnalyticsActions actions = new TextAnalyticsActions()
                {
                    AbstractiveSummarizeActions = new List<AbstractiveSummarizeAction>() { new AbstractiveSummarizeAction() },
                };

                // We'll get an ArgumentOutOfRangeException if the text is too short, so only enable this if we've crossed a certain threshold.
                if (text.Length > 40)
                {
                    actions.ExtractiveSummarizeActions = new List<ExtractiveSummarizeAction>() { new ExtractiveSummarizeAction() };
                }

                string[] documents = new string[] { text };
                AnalyzeActionsOperation operation = await _textClient.AnalyzeActionsAsync(WaitUntil.Completed, documents, actions);

                StringBuilder sb = new StringBuilder();
                await foreach (AnalyzeActionsResult result in operation.Value)
                {
                    foreach (AbstractiveSummarizeResult abstractResult in result.AbstractiveSummarizeResults.SelectMany(r => r.DocumentsResults))
                    {
                        if (abstractResult.HasError)
                        {
                            sb.AppendLine($"Abstractive summarization failed: {abstractResult.Error.ErrorCode} - {abstractResult.Error.Message}");
                            continue;
                        }

                        foreach (AbstractiveSummary summary in abstractResult.Summaries)
                        {
                            sb.AppendLine(summary.Text);
                        }
                    }

                    foreach (ExtractiveSummarizeResult extractResult in result.ExtractiveSummarizeResults.SelectMany(r => r.DocumentsResults))
                    {
                        if (extractResult.HasError)
                        {
                            sb.AppendLine($"Extractive summarization failed: {extractResult.Error.ErrorCode} - {extractResult.Error.Message}");
                            continue;
                        }

                        if (extractResult.Sentences.Count() > 0)
                        {
                            sb.AppendLine("Key sentences in the text:");

                            foreach (ExtractiveSummarySentence sentence in extractResult.Sentences)
                            {
                                sb.AppendLine(sentence.Text);
                            }
                        }
                    }
                }

                return sb.ToString();
            }
            catch (RequestFailedException ex)
            {
                return GetRequestFailedString(ex);
            }
            catch (ArgumentOutOfRangeException)
            {
                return "Text analysis failed due to the input text. This can happen when the provided text is too short.";
            }
        }

        private static string GetRequestFailedString(RequestFailedException ex) => ex.Status switch
        {
            401 => "Request failed due to authentication failure. The AI services key or endpoint may be misconfigured",
            429 => "The text could not be summarized because too many summarization requests have happened recently.",
            _ => $"Summarization failed: {ex.Message}",
        };

    }
}
