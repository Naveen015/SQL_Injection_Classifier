from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

MODEL_PATH = './sql_injection_model'

def predict_sql_input(text, model_path=MODEL_PATH):
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return "SQL Injection Detected" if prediction == 1 else "Safe Input"

# Example usage
if __name__ == "__main__":
    while True:
        query = input("Enter input text to classify (or type 'exit' to quit): ")
        
        if query.lower() in ["exit", "end"]:
            print("Exiting.")
            break

        if not query.strip():  # ignore empty input
            print("Please enter a valid query.")
            continue

        result = predict_sql_input(query)
        print(f"Prediction: {result}")
