from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(load_in_8bit=True)
torch.cuda.empty_cache()

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-2-12b-chat")

model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/stablelm-2-12b-chat",
    device_map="auto",
    torch_dtype=torch.float16
)

# Define system prompt and user message
system_prompt = """\
You are a helpful assistant with access to the following functions. You must use them if required -\n
"""

with open('data/functions.json','r') as f:
    system_prompt += f.read()

print(system_prompt)

messages = [
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': "Please, provide a quote for the following shipment: Container 40hc hazmat"}
]

# Prepare input text for the model
chat_input = f"{system_prompt}\n\nUser: {messages[1]['content']}\n\nAssistant:"
inputs = tokenizer(
    chat_input,
    return_tensors="pt",
    padding=True,
    truncation=True
).to(model.device)

# Generate tokens
tokens = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=100,  # Lower token limit
    temperature=0.7,     # Balance diversity and speed
    num_beams=2,          # Faster than sampling
    do_sample=True,      # Enable sampling for diversity
)

# Decode the output
output = tokenizer.decode(tokens[0], skip_special_tokens=True)
generated_response = output[len(chat_input):].strip()

print(generated_response)
