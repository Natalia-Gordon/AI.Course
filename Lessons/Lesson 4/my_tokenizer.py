from transformers import AutoTokenizer

tokenizer1 = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer2 = AutoTokenizer.from_pretrained("roberta-base") # מכיר עברית
tokenizer3 = AutoTokenizer.from_pretrained("xlm-roberta-base")

text = "איך נראית טוקניזציה בעברית?"

tokens = tokenizer1.tokenize(text)
# Get tokenizer ids
tokenizer_ids = tokenizer1.encode(text)

print(f"Original text: {text}")
print(f"Tokenizer IDs: {tokenizer_ids}")

# You can also get the tokens (the string representation of the IDs)
#tokens = tokenizer1.convert_ids_to_tokens(tokenizer1)
print(f"Tokens: {tokens}") 


#from model_name in models:
