from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier(
    "In Germany, how will Oktoberfest be?",
    candidate_labels=["festival", "politics", "economy", "weather"],
)
print(result)