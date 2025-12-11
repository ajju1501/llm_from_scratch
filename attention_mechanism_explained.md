# Understanding Weight Matrices in Self-Attention

## Code Breakdown

```python
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
```

---

## Line-by-Line Explanation

### 1. `torch.manual_seed(123)`
- **Purpose**: Sets the random seed for reproducibility
- **Why it matters**: Ensures that every time you run this code, the random weight matrices will have the **exact same values**
- **Value `123`**: Arbitrary seed number — you can use any integer

---

### 2-4. Creating Weight Matrices (W_query, W_key, W_value)

```python
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
```

Let's break this down piece by piece:

| Component | What it does |
|-----------|--------------|
| `torch.rand(d_in, d_out)` | Creates a tensor of shape `(d_in, d_out)` filled with **random values** uniformly distributed between 0 and 1 |
| `torch.nn.Parameter(...)` | Wraps the tensor as a **learnable parameter** (typically used in neural network modules) |
| `requires_grad=False` | **Disables gradient computation** — meaning these weights **won't be updated** during backpropagation |

---

## What are d_in and d_out?

From your notebook:
- **`d_in = 3`** → The input embedding dimension (each word is represented as a 3-dimensional vector)
- **`d_out = 2`** → The output dimension for queries, keys, and values (projecting to 2 dimensions)

So each weight matrix has shape **(3, 2)**:
```
W_query, W_key, W_value → Shape: (3, 2)
```

---

## The Role of Q, K, V in Attention

| Matrix | Role | Description |
|--------|------|-------------|
| **W_query** | Query Weights | Transforms input into **Query** vectors — "What am I looking for?" |
| **W_key** | Key Weights | Transforms input into **Key** vectors — "What do I contain?" |
| **W_value** | Value Weights | Transforms input into **Value** vectors — "What information do I provide?" |

---

## Visual Representation

```
Input Token (x)              Weight Matrices              Output
   Shape: (3,)      ×        Shape: (3, 2)        =      Shape: (2,)
   
   [0.55]                    [[w11, w12],                 [q1]
    0.87          ×           [w21, w22],         =       [q2]  ← Query
    0.66]                     [w31, w32]]

                    Same process for Key and Value
```

---

## Why `requires_grad=False`?

In this example, the weights are:
- **Randomly initialized**
- **NOT trainable** (static/frozen)

This is useful for:
1. **Learning purposes** — Understanding attention without training complexity
2. **Inference mode** — Using pre-trained weights without updating them

In a real transformer model, you would set `requires_grad=True` so the model can **learn optimal weight values** through backpropagation.

---

## Summary

| Aspect | Value/Description |
|--------|-------------------|
| Random Seed | `123` for reproducibility |
| Input Dimension | `d_in = 3` |
| Output Dimension | `d_out = 2` |
| Weight Matrix Shape | `(3, 2)` each |
| Trainable | No (`requires_grad=False`) |
| Purpose | Project input embeddings into Query, Key, Value spaces |

---

## What Happens Next?

After creating these weight matrices, you would:

1. **Compute Query**: `q = x @ W_query`
2. **Compute Key**: `k = x @ W_key`
3. **Compute Value**: `v = x @ W_value`
4. **Calculate Attention Scores**: `scores = q @ k.T`
5. **Apply Softmax**: `attention_weights = softmax(scores)`
6. **Compute Context Vector**: `context = attention_weights @ v`

This is the foundation of the **self-attention mechanism** used in Transformers!
