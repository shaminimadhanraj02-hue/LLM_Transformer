from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-cased")
result = unmasker("In Germany, Oktoberfest happens in this [MASK].", top_k=3)
for r in result:
    print(f"{r['sequence']}  (score={r['score']:.3f})")
