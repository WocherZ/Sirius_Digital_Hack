from transformers import BertTokenizerFast, AutoModelForSequenceClassification
from peft import PeftModel
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: ', device)

base_model_name = "blanchefort/rubert-base-cased-sentiment-rusentiment"
tokenizer = BertTokenizerFast.from_pretrained(base_model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name).to(device)

staying_possibility_adapter_name = "Vzvorygin/sirius_hack_staying_possibility"
staying_possibility_model = PeftModel.from_pretrained(base_model, staying_possibility_adapter_name).to(device)

recommendation_possibility_adapter_name = "Vzvorygin/sirius_hack_recommendation_possibility"
recommendation_possibility_model = PeftModel.from_pretrained(base_model, recommendation_possibility_adapter_name).to(device)

returning_possibility_adapter_name = "Vzvorygin/sirius_hack_returning_possibility"
returning_possibility_model = PeftModel.from_pretrained(base_model, returning_possibility_adapter_name).to(device)


def predict(text, model, tokenizer):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(predicted, dim=1).cpu().numpy()
    return predicted

def number_to_string(value):
  mapping = {
      0: "NEUTRAL",
      1: "POSITIVE",
      2: "NEGATIVE"
  }
  return mapping.get(value, "Invalid number")

def string_to_number(value):
  mapping = {
      "NEUTRAL": 0,
      "POSITIVE": 1,
      "NEGATIVE": 2
  }
  return mapping.get(value, "Invalid string")

def get_model_class_answer(text: str, model, tokenizer) -> str:
  predicted_class_label = predict(text, model, tokenizer)[0]
  return number_to_string(predicted_class_label)

def get_single_sentiment_analysis(text: str,
                                  type: str,
                                  base_model,
                                  tokenizer) -> dict:
    if type == 'Возможность остаться':
        get_model_class_answer(text, staying_possibility_model, tokenizer) 
    elif type == 'Возможность возвращения':
        get_model_class_answer(text, recommendation_possibility_model, tokenizer)           
    elif type == 'Рекомендация':
        get_model_class_answer(text, returning_possibility_model, tokenizer)
    else:
        get_model_class_answer(text, base_model, tokenizer)

def get_multi_sentiment_analysis(texts: list, type: str, model, tokenizer) -> dict:
    sentiment_count = {"NEUTRAL": 0, "POSITIVE": 0, "NEGATIVE": 0}

    for text in texts:
        predicted_santiment = get_single_sentiment_analysis(text, type, model, tokenizer)
        
        if predicted_santiment in sentiment_count:
           sentiment_count[predicted_santiment] += 1
        else:
            sentiment_count[predicted_santiment] = 1

    return sentiment_count

