from transformers import pipeline

# Load GPT-2 model for text generation
generator = pipeline("text-generation", model="distilgpt2")

# Define the prompt
prompt = "Their fresh bread is made in-house and it's amazing! Perfect for "

# Generate continuation
result = generator(prompt, truncation=True, max_length=100, do_sample=True)[0][
    "generated_text"
]

# Print the result
print(result)
