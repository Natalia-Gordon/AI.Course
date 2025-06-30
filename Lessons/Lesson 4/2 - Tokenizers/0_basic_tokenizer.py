from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Her unhappiness and playfulness were discussed during the reorganization of the department."
# text = "pseudopseudohypoparathyroidism"

# Tokenize into subword tokens
tokens = tokenizer.tokenize(text)

# Convert tokens to token IDs (input to the model)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

# Reconstruct text from token IDs
reconstructed = tokenizer.decode(token_ids)

print("Original text:")
print(text)
print("\nTokens:")
print(tokens)
print("\nToken IDs:")
print(token_ids)
print("\nReconstructed text:")
print(reconstructed)
