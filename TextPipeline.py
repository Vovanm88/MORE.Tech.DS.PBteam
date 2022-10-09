from transformers import MBartTokenizer, MBartForConditionalGeneration
from transformers import BertForMaskedLM,BertTokenizer, pipeline
import torch.utils.data as data
import torch
import torch.nn as nn
import numpy as np

#Summarization pipeline
class SummarizationPipeline:
    def __init__(self, DEVICE="cpu"):
        model_name = "IlyaGusev/mbart_ru_sum_gazeta"
        self.tokenizer = MBartTokenizer.from_pretrained(model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(DEVICE)
        self.device = DEVICE
    def __call__(self, text, max_len = 600):
        text = str(text)
        input_ids = self.tokenizer(
            [text],
            max_length = max_len,
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(self.device)

        output_ids = self.model.generate(input_ids=input_ids, no_repeat_ngram_size=4)[0]
        summary = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return summary
#Embedding Generation pipeline
class BertEmbeddingsPipeline:
    def __init__(self, DEVICE="cpu"):
        self.model=BertForMaskedLM.from_pretrained('sberbank-ai/ruBert-base')
        self.model.cls.predictions.decoder = nn.Identity()
        self.model = self.model.to(DEVICE)
        self.device = DEVICE
        self.tokenizer=BertTokenizer.from_pretrained('sberbank-ai/ruBert-base', do_lower_case=False)
    def __call__(self, text):
        input_ids = self.tokenizer(
            [text],
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(self.device)
        #print(self.model(input_ids).logits.shape)
        out = self.model(input_ids).logits
        out = torch.mean(out[0], dim=0)
        return out

class Classifier(nn.Module):
    def __init__(
            self, input_size=768, number_classes=3,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 2000),
            nn.GELU(),
            nn.Linear(2000, 1000),
            nn.GELU(),
            nn.Linear(1000, number_classes),
            #nn.Softmax(),
        )
    def forward(self, x):
        x = self.classifier(x)
        return x

class ClassificationPipeline:
    def __init__(self, path, DEVICE='cpu'):
        self.model = Classifier()
        loaded = self.model.load_state_dict(torch.load(path))
        self.model.to(DEVICE)
        self.DEVICE = DEVICE
        print(loaded)
    def __call__(self, input):
        self.model.eval()
        input = input.to(self.DEVICE)
        output = self.model(input).softmax()
        return output.detach().cpu().numpy()

class TextProcessingPipeline:
    def __init__(self, ClassificationPath, DEVICE):
        self.summarize = SummarizationPipeline(DEVICE)
        self.embedize = BertEmbeddingsPipeline(DEVICE)
        self.classificator = ClassificationPipeline(ClassificationPath, DEVICE)
    def __call__(self, text):
        summary_text = self.summarize(text)
        embs = self.embedize(summary_text)
        cat = self.classificator(embs)
        return summary_text, cat[0], cat[1], cat[2]

#pipe = TextProcessingPipeline("model.pt", 'cpu')
#out = pipe("text")

