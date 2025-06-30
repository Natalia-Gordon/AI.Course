from transformers import AutoTokenizer

models = ["bert-base-uncased", "roberta-base", "xlm-roberta-base"]

# text = "Unbelievable!"
text = "איך נראית tokenization בעברית?"

for model_name in models:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.tokenize(text)
    print(f"\n{model_name}:")
    print(tokens)
