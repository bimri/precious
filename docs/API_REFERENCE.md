# Precious-NLP Package API Reference

## Overview

The precious-nlp package provides three tokenizer-free approaches for natural language processing:

- **T-FREE**: A custom tokenizer-free encoding approach with word-level and character-level representations
- **CANINE**: Character-level processing with downsampling/upsampling mechanisms
- **Byte-level**: Direct byte-level text processing

## Quick Start

```python
# Import with hyphenated package name
import precious_nlp as precious
from precious_nlp import PreciousModel, PreciousConfig

# Create a model with T-FREE approach
config = precious.PreciousConfig(mode="tfree", d_model=256)
model = precious.PreciousModel(config)

# Process text
inputs = ["Hello world!", "This is a test."]
outputs = model(inputs)
logits = outputs["logits"]  # Shape: [batch_size, seq_len, vocab_size]
```

## Core Classes

### PreciousConfig

Configuration class for the Precious model.

```python
@dataclass
class PreciousConfig:
    mode: Literal["tfree", "canine", "byte"] = "tfree"
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    max_len: int = 1024
    # T-FREE specific parameters
    tfree_vocab_v: int = 8192
    tfree_m: int = 10
    tfree_k_lower: int = 0
    # CANINE specific parameters
    canine_K: int = 8
    canine_B: int = 16384
    canine_use_ngrams: bool = False
    canine_down_rate: int = 4
```

#### Parameters

- **mode** (`str`): Processing mode - "tfree", "canine", or "byte"
- **d_model** (`int`): Hidden dimension size
- **n_heads** (`int`): Number of attention heads
- **n_layers** (`int`): Number of transformer layers
- **max_len** (`int`): Maximum sequence length
- **tfree_vocab_v** (`int`): T-FREE vocabulary size
- **tfree_m** (`int`): Maximum word length for character encoding
- **tfree_k_lower** (`int`): Minimum frequency threshold for vocabulary
- **canine_K** (`int`): CANINE parameter K
- **canine_B** (`int`): CANINE vocabulary size
- **canine_use_ngrams** (`bool`): Whether to use n-grams in CANINE
- **canine_down_rate** (`int`): Downsampling rate for CANINE

#### Examples

```python
import precious_nlp as precious

# Small model for experimentation
small_config = precious.PreciousConfig(
    mode="byte",
    d_model=128,
    n_heads=4,
    n_layers=2
)

# Large model for production
large_config = precious.PreciousConfig(
    mode="tfree",
    d_model=768,
    n_heads=12,
    n_layers=12,
    tfree_vocab_v=16384
)

# CANINE configuration
canine_config = precious.PreciousConfig(
    mode="canine",
    d_model=512,
    canine_B=32768,
    canine_down_rate=8
)
```

### PreciousModel

Main model class implementing tokenizer-free transformer architecture.

```python
class PreciousModel(nn.Module):
    def __init__(self, config: PreciousConfig)
    def forward(self, inputs: List[str], targets: Optional[List[str]] = None) -> Dict[str, torch.Tensor]
```

#### Methods

##### `__init__(config: PreciousConfig)`

Initialize the model with the given configuration.

**Parameters:**
- **config**: PreciousConfig object specifying model architecture

**Example:**
```python
import precious_nlp as precious
config = precious.PreciousConfig(mode="byte", d_model=256)
model = precious.PreciousModel(config)
```

##### `forward(inputs: List[str], targets: Optional[List[str]] = None) -> Dict[str, torch.Tensor]`

Forward pass through the model.

**Parameters:**
- **inputs**: List of input text strings
- **targets**: Optional list of target strings for loss computation

**Returns:**
Dictionary containing:
- **logits**: Output logits tensor
- **loss**: Loss tensor (if targets provided)

**Examples:**

```python
# Inference only
inputs = ["Hello world", "Another example"]
outputs = model(inputs)
print(outputs["logits"].shape)  # [2, seq_len, vocab_size]

# Training with targets
inputs = ["Hello world", "Test sentence"]
targets = ["Hello world!", "Test sentence."]
outputs = model(inputs, targets=targets)
print(f"Loss: {outputs['loss']}")
print(f"Logits shape: {outputs['logits'].shape}")
```

## Mode-Specific Components

### T-FREE Mode

#### TFreeEncoder

Tokenizer-free encoder using word-level and character-level representations.

```python
class TFreeEncoder(nn.Module):
    def __init__(self, vocab_size_v: int, hidden_size: int, m: int, k_lower: int)
    def split_text(self, text: str) -> List[str]
    def build_vocabulary(self, texts: List[str])
    def forward(self, word_seqs: List[str]) -> torch.Tensor
    def word_indices(self, word: str) -> List[int]
```

**Key Features:**
- Automatic vocabulary building from text
- Character-level fallback for OOV words
- Bidirectional LSTM for character composition

**Example:**
```python
# Build vocabulary and encode text
encoder = TFreeEncoder(vocab_size_v=8192, hidden_size=512, m=20, k_lower=2)
texts = ["Hello world", "This is training data", "More examples"]
encoder.build_vocabulary(texts)

words = encoder.split_text("Hello world")
embeddings = encoder(words)
print(embeddings.shape)  # [seq_len, hidden_size]
```

#### TFreeMLHead

Multi-label prediction head for T-FREE models.

```python
class TFreeMLHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size_v: int)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

### CANINE Mode

#### CanineEmbedding

Character-level embedding following CANINE approach.

```python
class CanineEmbedding(nn.Module):
    def __init__(self, d: int, K: int, B: int, use_ngrams: bool = False)
    def forward(self, x: str) -> torch.Tensor
```

**Example:**
```python
embedding = CanineEmbedding(d=512, K=8, B=16384)
text = "Hello"
char_embeddings = embedding(text)
print(char_embeddings.shape)  # [5, 512] - one embedding per character
```

#### CanineDownUp

Downsampling and upsampling module for sequence length reduction.

```python
class CanineDownUp(nn.Module):
    def __init__(self, d: int, down_rate: int)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

### Byte Mode

In byte mode, text is processed directly at the byte level using standard embedding layers.

**Example:**
```python
import precious_nlp as precious
config = precious.PreciousConfig(mode="byte", d_model=256)
model = precious.PreciousModel(config)

# Each character is converted to its byte representation
inputs = ["Hello 🌍"]  # Works with Unicode
outputs = model(inputs)
```

### EVAAttention

Enhanced Vanilla Attention mechanism used across all modes.

```python
class EVAAttention(nn.Module):
    def __init__(self, dropout: float = 0.1, causal: bool = False)
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor
```

**Features:**
- Optional causal masking for autoregressive tasks
- Dropout for regularization
- Efficient attention computation

## Training Examples

### Basic Training Loop

```python
import torch
from torch.optim import AdamW
import precious_nlp as precious
from precious_nlp import PreciousModel, PreciousConfig

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
config = precious.PreciousConfig(mode="byte", d_model=256, n_layers=4)
model = precious.PreciousModel(config).to(device)
optimizer = AdamW(model.parameters(), lr=1e-4)
```

# Training data
train_inputs = ["This is input text", "Another example"]
train_targets = ["This is target text", "Another target"]

# Training step
model.train()
optimizer.zero_grad()
outputs = model(train_inputs, targets=train_targets)
loss = outputs["loss"]
loss.backward()
optimizer.step()

print(f"Training loss: {loss.item()}")
```

### Advanced Training with Validation

```python
def train_epoch(model, train_data, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_inputs, batch_targets in train_data:
        optimizer.zero_grad()
        outputs = model(batch_inputs, targets=batch_targets)
        
        if "loss" in outputs:
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return total_loss / len(train_data)

def validate(model, val_data, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_inputs, batch_targets in val_data:
            outputs = model(batch_inputs, targets=batch_targets)
            if "loss" in outputs:
                total_loss += outputs["loss"].item()
    
    return total_loss / len(val_data)

# Training loop
for epoch in range(10):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss = validate(model, val_loader, device)
    print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")
```

## Performance Considerations

### Memory Usage

Different modes have different memory characteristics:

- **Byte mode**: Most memory efficient, direct byte processing
- **CANINE mode**: Medium memory usage, character-level processing
- **T-FREE mode**: Highest memory usage due to dual representations

### Sequence Length Scaling

All modes use transformer architecture, so memory scales quadratically with sequence length.

```python
# For long sequences, consider chunking
def process_long_text(model, text, chunk_size=512):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    results = []
    
    for chunk in chunks:
        output = model([chunk])
        results.append(output["logits"])
    
    return torch.cat(results, dim=1)
```

### Batch Processing

Models support variable-length batches efficiently:

```python
# Variable length inputs are handled automatically
inputs = [
    "Short text",
    "This is a much longer text with many more tokens",
    "Medium length"
]
outputs = model(inputs)  # Automatically padded and processed
```

## Error Handling

### Common Issues and Solutions

1. **CUDA out of memory**: Reduce batch size or model size
2. **Empty input strings**: Model handles empty inputs gracefully
3. **Unicode characters**: All modes support Unicode text

```python
try:
    outputs = model(inputs)
except RuntimeError as e:
    if "out of memory" in str(e):
        # Reduce batch size and retry
        smaller_batches = [inputs[i:i+2] for i in range(0, len(inputs), 2)]
        outputs = [model(batch) for batch in smaller_batches]
    else:
        raise e
```

## Migration Guide

### From Other Tokenizer-based Models

```python
# Instead of tokenizer-based approach:
# tokenizer = AutoTokenizer.from_pretrained("model")
# tokens = tokenizer(texts, return_tensors="pt")
# outputs = model(**tokens)

# Use precious-nlp directly:
import precious_nlp as precious
model = precious.PreciousModel(precious.PreciousConfig(mode="byte"))
outputs = model(texts)
```

### Choosing the Right Mode

- **Use byte mode** for: General purpose, Unicode support, memory efficiency
- **Use CANINE mode** for: Character-aware tasks, multilingual processing
- **Use T-FREE mode** for: Vocabulary-aware tasks, research on tokenizer-free methods

## Troubleshooting

### Performance Issues

1. **Slow inference**: Try smaller model dimensions or fewer layers
2. **High memory usage**: Use byte mode or reduce batch size
3. **Poor convergence**: Adjust learning rate or try different modes

### Debugging Tips

```python
# Check model outputs
outputs = model(["test"])
print(f"Output keys: {outputs.keys()}")
print(f"Logits shape: {outputs['logits'].shape}")
print(f"Device: {outputs['logits'].device}")

# Verify gradient flow
loss = outputs.get("loss")
if loss is not None:
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"No gradient for: {name}")
```

For more examples and tutorials, see the `tests/` directory in the repository.