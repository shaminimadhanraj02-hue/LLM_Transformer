from transformers import pipeline

ner = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",   
)

text = "I am Shamini studying masters at FAU in Erlangen."
result = ner(text)


for ent in result:
    print(f"{ent['word']} → {ent['entity_group']} (score={float(ent['score']):.3f})")
