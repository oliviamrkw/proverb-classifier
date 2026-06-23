from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Download happens automatically on first run
model_name = "bigscience/bloomz-7b1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True  # Use 4-bit quantization to fit in 8GB RAM
)

# Test
prompt = "Classify this as figurative (1) or literal (0): A barking dog never bites."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
result = tokenizer.decode(outputs[0])
print(result)