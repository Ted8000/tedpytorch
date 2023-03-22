from transformers import AutoModelForSequenceClassification

def get_model(path="xlm-roberta-base", num_labels=2):
    classify_model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=num_labels)
    return classify_model