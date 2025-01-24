# Training Transformers using SmolLM2-135 Model

## Model Architecture
The SmolLM2-135 architecture consists of the following components:

1. **Embedding Layer**:
   - Token embeddings: Maps vocabulary tokens to a dense vector representation.
   - Positional embeddings: Encodes positional information into the input sequence.

2. **Transformer Blocks**:
   - A stack of Transformer blocks, each containing:
     - **LayerNorm**: Normalizes the input to stabilize training.
     - **Causal Self-Attention**: Computes attention scores, applying a causal mask to prevent future token leakage.
     - **MLP**: A feedforward neural network with a single hidden layer.

3. **Output Layer**:
   - A fully connected layer that maps the hidden states to vocabulary logits for prediction.

## Model Configuration
| Parameter         | Value |
|-------------------|-------|
| Block size        | 1024  |
| Vocabulary size   | 50257 |
| Number of layers  | 8     |
| Number of heads   | 8     |
| Embedding size    | 512   |

### Parameter Calculation
The total number of parameters in the SmolLM2-135 model is:

1. **Embedding Parameters**:
   - Token embedding: `vocab_size * n_embd = 50257 * 512 = 25,766,656`
   - Positional embedding: `block_size * n_embd = 1024 * 512 = 524,288`

   **Total Embedding Parameters**: `25,766,656 + 524,288 = 26,290,944`

2. **Transformer Block Parameters** (per block):
   - Self-Attention:
     - Query, Key, Value projections: `n_embd * 3 * n_embd = 512 * 3 * 512 = 786,432`
     - Output projection: `n_embd * n_embd = 512 * 512 = 262,144`
   - MLP:
     - First linear layer: `n_embd * 4 * n_embd = 512 * 4 * 512 = 1,048,576`
     - Second linear layer: `4 * n_embd * n_embd = 2048 * 512 = 1,048,576`

   **Total per Block**: `786,432 + 262,144 + 1,048,576 + 1,048,576 = 3,145,728`

   With 8 blocks: `3,145,728 * 8 = 25,165,824`

3. **Final LayerNorm and Output Layer**:
   - LayerNorm: `2 * n_embd = 2 * 512 = 1,024`
   - Output layer: `n_embd * vocab_size = 512 * 50257 = 25,706,944`

   **Total Final Layer Parameters**: `1,024 + 25,706,944 = 25,707,968`

### Total Parameters
The total number of parameters in the SmolLM2-135 model:

```
Embedding Parameters       = 26,290,944
Transformer Block (8 blocks) = 25,165,824
Final Layer Parameters     = 25,707,968
------------------------------------------
Total Parameters           = 77,164,736
```

## Training
### Steps:
1. Load the input text file and tokenize it.
2. Train the model using batches of data.
3. Save the model checkpoint after training.
4. Optionally load a saved checkpoint to continue training.

### Features:
- Predicts outputs every 500 training steps.
- Saves checkpoints for resuming training later.
- Allows optional continuation of training based on user input.

### Result
- Since the Model was run on CoLab, had to reduce the number of steps to 200 and predict the output after every 20 steps.
- It creaters a Checkpoint after 200 steps
- Prompts the user for Resume from the Checkpont : Yes / No
- If "Yes", proceeds with training for next 50 steps with prediction every 10 steps.

