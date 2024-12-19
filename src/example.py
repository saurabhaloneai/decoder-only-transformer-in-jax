import jax
import jax.numpy as jnp
from jax import random
from src.gpt_2 import init_model, model_forward  
from transformers import AutoModelForCausalLM, AutoTokenizer

# tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# in
prompt = "everyone is retard but "
input_ids = tokenizer.encode(prompt, add_special_tokens=False)
input_ids = jnp.array(input_ids).reshape(1, -1)  # Batch size 1

# model hyperparameters (GPT-2 small)
vocab_size = tokenizer.vocab_size  # 50257
max_len = 1024
d_model = 768
num_heads = 12
num_layers = 12
d_ff = 3072

# Init model parameters
rng = random.PRNGKey(42)
params = init_model(rng, vocab_size, max_len, d_model, num_heads, num_layers, d_ff)
logits = model_forward(params, input_ids, num_heads, num_layers)

# preds
last_logits = logits[:, -1, :]  
next_token = jnp.argmax(last_logits, axis=-1)  
predicted_token = tokenizer.decode(int(next_token.item()))  

print(f"prompt: {prompt}")
print(f"predicted next token: {predicted_token}")

