using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.IO;
namespace TranslatorText
{
    public class TranslationResult
    {
        public DetectedLanguage DetectedLanguage { get; set; }
        public TextResult SourceText { get; set; }
        public Translation[] Translations { get; set; }
    }

    public class DetectedLanguage
    {
        public string Language { get; set; }
        public float Score { get; set; }
    }

    public class TextResult
    {
        public string Text { get; set; }
        public string Script { get; set; }
    }

    public class Translation
    {
        public string Text { get; set; }
        public TextResult Transliteration { get; set; }
        public string To { get; set; }
        public Alignment Alignment { get; set; }
        public SentenceLength SentLen { get; set; }
    }

    public class Alignment
    {
        public string Proj { get; set; }
    }

    public class SentenceLength
    {
        public int[] SrcSentLen { get; set; }
        public int[] TransSentLen { get; set; }
    }

    class Program
    {

        static public async Task TranslateTextRequest(string subscriptionKey, string region, string endpoint, string route)
        {
            string text = System.IO.File.ReadAllText(@"C:\Users\Dell\Downloads\frData.tsv");//read from a file
            char[] whitespace = new char[] { '\n' };
            string[] ssizes = text.Split(whitespace);
            //string resfinal = "AA   AA  AA \n";
            //string resfinal = "";
            for (int j = 13288; j <= 15000; j = j + 1)
            // for (int j = 100; j<=150;j=j+1)
            {
                Console.WriteLine(j);
                string inputText = ssizes[j];
                char[] whitespace1 = new char[] { '\t' };
                string[] final = inputText.Split(whitespace1);
                inputText = final[0];
               // Console.WriteLine(inputText);

                object[] body = new object[] { new { Text = inputText } };
                var requestBody = JsonConvert.SerializeObject(body);
                using (var client = new HttpClient())
                using (var request = new HttpRequestMessage())
                {

                    request.Method = HttpMethod.Post;
                    request.RequestUri = new Uri(endpoint + route);
                    request.Content = new StringContent(requestBody, Encoding.UTF8, "application/json");
                    request.Headers.Add("Ocp-Apim-Subscription-Key", subscriptionKey);
                    request.Headers.Add("Ocp-Apim-Subscription-Region", region);


                    HttpResponseMessage response = await client.SendAsync(request).ConfigureAwait(false);

                    string result = await response.Content.ReadAsStringAsync();

                    TranslationResult[] deserializedOutput = JsonConvert.DeserializeObject<TranslationResult[]>(result);

                    foreach (TranslationResult o in deserializedOutput)
                    {
                        if (o.DetectedLanguage.Language == "en")

                        {
                            Console.WriteLine("en");
                            continue;
                        }

                        
                        Console.WriteLine("fr");
                        foreach (Translation t in o.Translations)
                        {
                           
                            string a = inputText + "\n";
                            string b = t.Text + "\n";
                            string c = t.Alignment.Proj + "\n";
                            string res = a + b + c;
                            //resfinal = resfinal + res;
                            File.AppendAllText(@"C:\Users\Dell\Downloads\FrenchDataTrain.txt", res);
                        }
                    }
                }
            }
        }

        static async Task Main(string[] args)
        {
            // This is our main function.
            // Output languages are defined in the route.
            // For a complete list of options, see API reference.
            // https://docs.microsoft.com/azure/cognitive-services/translator/reference/v3-0-translate
            string route = "/translate?api-version=3.0&to=en&includeAlignment=true";
            // Prompts you for text to translate. If you'd prefer, you can
            // provide a string as textToTranslate.
            //Console.Write("Type the phrase you'd like to translate? ");
            //string textToTranslate = Console.ReadLine();
            await TranslateTextRequest("acdfacd6642b415b84727be466cc997b", "centralindia", "https://api.cognitive.microsofttranslator.com/", route);
            Console.WriteLine("Press any key to continue.");
            Console.ReadKey();
        }
    }
}
