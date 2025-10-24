from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"  # explicit model
)

result = classifier(
    "The Germany's weather is so unpredictable",
    candidate_labels=["education", "climate", "confused", "culture"],
)

print(result)
