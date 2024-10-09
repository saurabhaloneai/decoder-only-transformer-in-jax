import jax
import jax.numpy as jnp
from jax import random

def xavier_init(rng, shape):
    in_dim, out_dim = shape
    std = jnp.sqrt(2.0 / (in_dim + out_dim))
    return std * random.normal(rng, shape)

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    normalized = (x - mean) / jnp.sqrt(variance + eps)
    return gamma * normalized + beta

def init_embeddings(rng, vocab_size, d_model, max_len):
    rng_token, rng_pos = random.split(rng)
    token_embedding = xavier_init(rng_token, (vocab_size, d_model))
    positional_embedding = xavier_init(rng_pos, (max_len, d_model))
    return token_embedding, positional_embedding

def embed_tokens(token_embedding, input_ids):
    return token_embedding[input_ids]

def add_positional_embeddings(x, pos_emb):
    seq_len = x.shape[1]
    return x + pos_emb[:seq_len, :]

def split_heads(x, num_heads):
    batch_size, seq_len, d_model = x.shape
    depth = d_model // num_heads
    return x.reshape(batch_size, seq_len, num_heads, depth).transpose(0, 2, 1, 3)

def multi_head_attention(params, x, num_heads):
    q = jnp.matmul(x, params['W_q'])
    k = jnp.matmul(x, params['W_k'])
    v = jnp.matmul(x, params['W_v'])
    
    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    v = split_heads(v, num_heads)
    
    dk = q.shape[-1]
    scores = jnp.matmul(q, k.transpose(0,1,3,2)) / jnp.sqrt(dk)
    attn_weights = jax.nn.softmax(scores, axis=-1)
    attn_output = jnp.matmul(attn_weights, v)
    
    attn_output = attn_output.transpose(0,2,1,3).reshape(x.shape[0], x.shape[1], -1)
    output = jnp.matmul(attn_output, params['W_o'])
    return output

def feed_forward_network(params, x):
    x = jnp.matmul(x, params['W1']) + params['b1']
    x = jax.nn.relu(x)
    x = jnp.matmul(x, params['W2']) + params['b2']
    return x

def init_transformer_block(rng, d_model, num_heads, d_ff):
    rng_attn, rng_ff = random.split(rng)
    params = {
        'W_q': xavier_init(rng_attn, (d_model, d_model)),
        'W_k': xavier_init(rng_attn, (d_model, d_model)),
        'W_v': xavier_init(rng_attn, (d_model, d_model)),
        'W_o': xavier_init(rng_attn, (d_model, d_model)),
        'W1': xavier_init(rng_ff, (d_model, d_ff)),
        'b1': jnp.zeros((d_ff,)),
        'W2': xavier_init(rng_ff, (d_ff, d_model)),
        'b2': jnp.zeros((d_model,))
    }
    params['ln1_gamma'] = jnp.ones((d_model,))
    params['ln1_beta'] = jnp.zeros((d_model,))
    params['ln2_gamma'] = jnp.ones((d_model,))
    params['ln2_beta'] = jnp.zeros((d_model,))
    return params

def transformer_block(params, x, num_heads, d_ff):
    ln1 = layer_norm(x, params['ln1_gamma'], params['ln1_beta'])
    attn_output = multi_head_attention(params, ln1, num_heads)
    x = x + attn_output  
    ln2 = layer_norm(x, params['ln2_gamma'], params['ln2_beta'])
    ff_output = feed_forward_network(params, ln2)
    x = x + ff_output  
    return x

def init_model(rng, vocab_size, max_len, d_model, num_heads, num_layers, d_ff):
    token_emb, pos_emb = init_embeddings(rng, vocab_size, d_model, max_len)
    rng_blocks = random.split(rng, num_layers)
    transformer_blocks = [init_transformer_block(rng_blocks[i], d_model, num_heads, d_ff) for i in range(num_layers)]
    W_out = xavier_init(rng, (d_model, vocab_size))
    return {
        'token_emb': token_emb,
        'pos_emb': pos_emb,
        'transformer_blocks': transformer_blocks,
        'W_out': W_out
    }

def model_forward(params, input_ids, num_heads, num_layers):
    x = embed_tokens(params['token_emb'], input_ids)
    x = add_positional_embeddings(x, params['pos_emb'])
    for i in range(num_layers):
        x = transformer_block(params['transformer_blocks'][i], x, num_heads, d_ff=3072)
    logits = jnp.matmul(x, params['W_out'])
    return logits
