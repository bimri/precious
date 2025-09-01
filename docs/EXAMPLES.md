# Precious Package Examples and Tutorials

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage Examples](#basic-usage-examples)
3. [Advanced Training Examples](#advanced-training-examples)
4. [Mode Comparison](#mode-comparison)
5. [Custom Training Loops](#custom-training-loops)
6. [Performance Optimization](#performance-optimization)
7. [Real-world Applications](#real-world-applications)

## Getting Started

### Installation and Basic Setup

```python
# Install the package (when available on PyPI)
# pip install precious

# For development installation
import sys
sys.path.append('/path/to/precious')

from precious import PreciousModel, PreciousConfig
import torch

# Check device availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

### First Model

```python
# Create a simple byte-level model
config = PreciousConfig(
    mode="byte",
    d_model=256,
    n_heads=8,
    n_layers=4
)

model = PreciousModel(config).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test with simple input
inputs = ["Hello, tokenizer-free world!"]
outputs = model(inputs)
print(f"Output shape: {outputs['logits'].shape}")
```

## Basic Usage Examples

### Example 1: Text Classification Setup

```python
import torch
import torch.nn as nn
from precious import PreciousModel, PreciousConfig

class PreciousClassifier(nn.Module):
    """Text classifier using Precious backbone"""
    
    def __init__(self, config, num_classes):
        super().__init__()
        self.backbone = PreciousModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.d_model // 2, num_classes)
        )
    
    def forward(self, texts):
        # Get backbone outputs
        outputs = self.backbone(texts)
        logits = outputs["logits"]  # [batch, seq_len, d_model]
        
        # Pool over sequence length (mean pooling)
        pooled = logits.mean(dim=1)  # [batch, d_model]
        
        # Classification
        return self.classifier(pooled)

# Usage
config = PreciousConfig(mode="byte", d_model=256)
classifier = PreciousClassifier(config, num_classes=2)

# Sample classification task
texts = [
    "I love this movie!",
    "This film was terrible.",
    "Great acting and storyline.",
    "Boring and predictable."
]
labels = torch.tensor([1, 0, 1, 0])  # 1=positive, 0=negative

predictions = classifier(texts)
print(f"Predictions shape: {predictions.shape}")
```

### Example 2: Language Modeling

```python
def train_language_model():
    # Configuration for language modeling
    config = PreciousConfig(
        mode="tfree",
        d_model=512,
        n_heads=8,
        n_layers=6,
        tfree_vocab_v=16384
    )
    
    model = PreciousModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Sample training data
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming how we process text.",
        "Tokenizer-free approaches offer new possibilities.",
        "Natural language processing continues to evolve rapidly."
    ]
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # For language modeling, we use text as both input and target
    # The model learns to predict the next token
    outputs = model(texts, targets=texts)
    
    if "loss" in outputs:
        loss = outputs["loss"]
        print(f"Language modeling loss: {loss.item():.4f}")
        loss.backward()
        optimizer.step()
    
    return model

# Train the model
lm_model = train_language_model()
```

### Example 3: Text Generation

```python
def generate_text(model, prompt, max_length=50, temperature=1.0):
    """Simple text generation using byte-level model"""
    model.eval()
    
    # Convert prompt to bytes
    current_text = prompt
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            outputs = model([current_text])
            logits = outputs["logits"][0, -1, :]  # Last token logits
            
            # Apply temperature scaling
            logits = logits / temperature
            
            # Sample next byte
            probs = torch.softmax(logits, dim=-1)
            next_byte = torch.multinomial(probs, 1).item()
            
            # Convert back to character and append
            try:
                next_char = chr(next_byte)
                current_text += next_char
                
                # Stop at sentence end
                if next_char in '.!?':
                    break
            except ValueError:
                # Invalid byte value, skip
                continue
    
    return current_text

# Example usage with byte model
config = PreciousConfig(mode="byte", d_model=256, n_layers=4)
gen_model = PreciousModel(config)

# Generate text (Note: this requires a trained model for good results)
generated = generate_text(gen_model, "The future of AI is", max_length=30)
print(f"Generated: {generated}")
```

## Advanced Training Examples

### Example 4: Multi-Task Learning

```python
class MultiTaskPreciousModel(nn.Module):
    """Multi-task model using Precious backbone"""
    
    def __init__(self, config, task_configs):
        super().__init__()
        self.backbone = PreciousModel(config)
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, task_config in task_configs.items():
            self.task_heads[task_name] = nn.Linear(
                config.d_model, task_config["output_dim"]
            )
    
    def forward(self, texts, task=None):
        # Get backbone representations
        outputs = self.backbone(texts)
        pooled = outputs["logits"].mean(dim=1)  # Pool over sequence
        
        if task is None:
            # Return all task predictions
            return {
                task_name: head(pooled) 
                for task_name, head in self.task_heads.items()
            }
        else:
            # Return specific task prediction
            return self.task_heads[task](pooled)

# Setup multi-task model
task_configs = {
    "sentiment": {"output_dim": 3},  # positive, negative, neutral
    "topic": {"output_dim": 5},      # 5 topic categories
    "spam": {"output_dim": 2}        # spam, not spam
}

config = PreciousConfig(mode="canine", d_model=384)
multi_model = MultiTaskPreciousModel(config, task_configs)

# Training with multiple tasks
def train_multitask_step(model, texts, labels_dict, optimizer):
    model.train()
    optimizer.zero_grad()
    
    predictions = model(texts)
    total_loss = 0
    
    for task_name, task_labels in labels_dict.items():
        task_pred = predictions[task_name]
        task_loss = nn.CrossEntropyLoss()(task_pred, task_labels)
        total_loss += task_loss
        print(f"{task_name} loss: {task_loss.item():.4f}")
    
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

# Example training data
texts = ["This is a great product!", "Spam email content", "News article text"]
labels_dict = {
    "sentiment": torch.tensor([2, 1, 1]),  # positive, negative, negative
    "topic": torch.tensor([0, 3, 4]),     # product, spam, news
    "spam": torch.tensor([0, 1, 0])       # not spam, spam, not spam
}

optimizer = torch.optim.AdamW(multi_model.parameters(), lr=1e-4)
loss = train_multitask_step(multi_model, texts, labels_dict, optimizer)
print(f"Total multi-task loss: {loss:.4f}")
```

### Example 5: Curriculum Learning

```python
def curriculum_training():
    """Implement curriculum learning with increasing difficulty"""
    
    config = PreciousConfig(mode="byte", d_model=256, n_layers=4)
    model = PreciousModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    # Define curriculum stages
    curriculum_stages = [
        {
            "name": "Simple sentences",
            "data": [
                "The cat sits.",
                "Dogs run fast.",
                "Birds fly high.",
                "Fish swim deep."
            ],
            "epochs": 5
        },
        {
            "name": "Complex sentences", 
            "data": [
                "The quick brown fox jumps over the lazy dog.",
                "Scientists discovered a new species in the deep ocean.",
                "Technology continues to advance at an unprecedented pace.",
                "Climate change affects ecosystems around the world."
            ],
            "epochs": 10
        },
        {
            "name": "Long paragraphs",
            "data": [
                "Natural language processing has revolutionized how we interact with computers. " +
                "From simple keyword matching to sophisticated transformer models, the field has " +
                "evolved dramatically over the past few decades.",
                
                "The development of tokenizer-free approaches represents a new frontier in NLP. " +
                "By processing text at the character or byte level, these methods avoid the " +
                "limitations of traditional tokenization schemes."
            ],
            "epochs": 15
        }
    ]
    
    # Train through curriculum stages
    for stage in curriculum_stages:
        print(f"\n=== Training Stage: {stage['name']} ===")
        
        for epoch in range(stage["epochs"]):
            model.train()
            total_loss = 0
            
            for text in stage["data"]:
                optimizer.zero_grad()
                outputs = model([text], targets=[text])
                
                if "loss" in outputs:
                    loss = outputs["loss"]
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            avg_loss = total_loss / len(stage["data"])
            print(f"Epoch {epoch+1}/{stage['epochs']}, Loss: {avg_loss:.4f}")
            
            # Update learning rate
            scheduler.step(avg_loss)
    
    return model

# Run curriculum learning
curriculum_model = curriculum_training()
```

## Mode Comparison

### Example 6: Comparing All Three Modes

```python
def compare_modes():
    """Compare performance and characteristics of all three modes"""
    
    # Same architecture for fair comparison
    base_config = {
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 4
    }
    
    configs = [
        PreciousConfig(mode="tfree", **base_config, tfree_vocab_v=4096),
        PreciousConfig(mode="canine", **base_config, canine_B=8192),
        PreciousConfig(mode="byte", **base_config)
    ]
    
    # Test texts with different characteristics
    test_texts = [
        "Simple English text.",
        "Text with numbers: 12345 and symbols: @#$%",
        "Multiple languages: Hello, Bonjour, Hola, こんにちは",
        "Very long text: " + "word " * 100
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n=== Testing {config.mode.upper()} mode ===")
        
        model = PreciousModel(config).to(device)
        mode_results = {}
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        mode_results["parameters"] = total_params
        print(f"Parameters: {total_params:,}")
        
        # Test inference time
        model.eval()
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if torch.cuda.is_available():
            start_time.record()
        
        with torch.no_grad():
            for text in test_texts:
                outputs = model([text])
                print(f"Input: '{text[:30]}...' -> Output shape: {outputs['logits'].shape}")
        
        if torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            mode_results["inference_time_ms"] = elapsed_time
            print(f"Total inference time: {elapsed_time:.2f}ms")
        
        # Test training step
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        train_start = time.time()
        optimizer.zero_grad()
        outputs = model(test_texts[:2], targets=test_texts[:2])
        if "loss" in outputs:
            outputs["loss"].backward()
            optimizer.step()
            mode_results["training_loss"] = outputs["loss"].item()
            print(f"Training loss: {outputs['loss'].item():.4f}")
        train_end = time.time()
        
        mode_results["training_time_ms"] = (train_end - train_start) * 1000
        print(f"Training step time: {(train_end - train_start)*1000:.2f}ms")
        
        results[config.mode] = mode_results
    
    # Summary comparison
    print("\n=== COMPARISON SUMMARY ===")
    print(f"{'Mode':<10} {'Params':<12} {'Inf.(ms)':<10} {'Train(ms)':<12} {'Loss':<8}")
    print("-" * 60)
    
    for mode, result in results.items():
        print(f"{mode.upper():<10} {result['parameters']:<12,} "
              f"{result.get('inference_time_ms', 0):<10.1f} "
              f"{result.get('training_time_ms', 0):<12.1f} "
              f"{result.get('training_loss', 0):<8.4f}")
    
    return results

# Run comparison
import time
comparison_results = compare_modes()
```

## Custom Training Loops

### Example 7: Advanced Training with Validation and Checkpointing

```python
import os
from pathlib import Path
import json

class PreciousTrainer:
    """Advanced trainer with validation and checkpointing"""
    
    def __init__(self, model, train_data, val_data, config):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 0.01)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.get("epochs", 100)
        )
        
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_inputs, batch_targets in self.train_data:
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_inputs, targets=batch_targets)
            
            if "loss" in outputs:
                loss = outputs["loss"]
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in self.val_data:
                outputs = self.model(batch_inputs, targets=batch_targets)
                
                if "loss" in outputs:
                    total_loss += outputs["loss"].item()
                    num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "config": self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"], checkpoint["val_loss"]
    
    def train(self, epochs):
        """Main training loop"""
        print(f"Starting training for {epochs} epochs...")
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate()
            val_losses.append(val_loss)
            
            # Learning rate scheduling
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Logging
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"LR: {current_lr:.2e}")
            
            # Checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % 5 == 0:  # Save every 5 epochs
                self.save_checkpoint(epoch + 1, val_loss, is_best)
        
        # Save final results
        results = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": self.best_val_loss
        }
        
        with open(self.checkpoint_dir / "training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results

# Usage example
def prepare_dummy_data():
    """Prepare dummy training data"""
    train_texts = [
        "Training example one with some content.",
        "Another training example for the model.",
        "More training data to learn from.",
        "Additional examples for better learning."
    ]
    
    val_texts = [
        "Validation example for evaluation.",
        "Another validation sample."
    ]
    
    # Create batched data (simplified)
    train_data = [(train_texts, train_texts)]  # Self-supervised
    val_data = [(val_texts, val_texts)]
    
    return train_data, val_data

# Setup and train
config = PreciousConfig(mode="byte", d_model=256, n_layers=4)
model = PreciousModel(config).to(device)

train_data, val_data = prepare_dummy_data()

trainer_config = {
    "lr": 1e-4,
    "weight_decay": 0.01,
    "epochs": 20,
    "checkpoint_dir": "./precious_checkpoints"
}

trainer = PreciousTrainer(model, train_data, val_data, trainer_config)
training_results = trainer.train(epochs=20)

print(f"Training completed. Best validation loss: {training_results['best_val_loss']:.4f}")
```

## Performance Optimization

### Example 8: Memory-Efficient Training

```python
def memory_efficient_training():
    """Demonstrate memory optimization techniques"""
    
    # Use smaller model for memory efficiency
    config = PreciousConfig(
        mode="byte",
        d_model=128,  # Smaller hidden size
        n_heads=4,    # Fewer attention heads
        n_layers=3    # Fewer layers
    )
    
    model = PreciousModel(config).to(device)
    
    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Use mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    def train_step_with_amp(inputs, targets):
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs, targets=targets)
                loss = outputs.get("loss")
            
            if loss is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                return loss.item()
        else:
            # CPU training without mixed precision
            outputs = model(inputs, targets=targets)
            loss = outputs.get("loss")
            if loss is not None:
                loss.backward()
                optimizer.step()
                return loss.item()
        
        return 0
    
    # Training with memory monitoring
    if torch.cuda.is_available():
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    
    # Batch processing with smaller chunks
    large_texts = ["Sample text " * 50] * 8  # Large batch
    chunk_size = 2
    
    total_loss = 0
    for i in range(0, len(large_texts), chunk_size):
        chunk = large_texts[i:i+chunk_size]
        loss = train_step_with_amp(chunk, chunk)
        total_loss += loss
        
        if torch.cuda.is_available() and i % 4 == 0:
            print(f"Chunk {i//chunk_size + 1}: GPU memory: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    
    print(f"Average loss: {total_loss / (len(large_texts) // chunk_size):.4f}")
    
    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Final GPU memory: {torch.cuda.memory_allocated()/1e6:.1f} MB")

# Run memory-efficient training
memory_efficient_training()
```

## Real-world Applications

### Example 9: Document Similarity

```python
class DocumentSimilarity:
    """Document similarity using Precious embeddings"""
    
    def __init__(self, model_config=None):
        if model_config is None:
            model_config = PreciousConfig(mode="byte", d_model=384, n_layers=6)
        
        self.model = PreciousModel(model_config).to(device)
        self.model.eval()
    
    def get_document_embedding(self, text):
        """Get document-level embedding"""
        with torch.no_grad():
            outputs = self.model([text])
            # Average pooling over sequence dimension
            embedding = outputs["logits"].mean(dim=1).squeeze(0)  # [d_model]
            return embedding
    
    def compute_similarity(self, text1, text2):
        """Compute cosine similarity between two texts"""
        emb1 = self.get_document_embedding(text1)
        emb2 = self.get_document_embedding(text2)
        
        # Cosine similarity
        similarity = torch.cosine_similarity(emb1, emb2, dim=0)
        return similarity.item()
    
    def find_most_similar(self, query, documents):
        """Find most similar document to query"""
        query_emb = self.get_document_embedding(query)
        
        similarities = []
        for doc in documents:
            doc_emb = self.get_document_embedding(doc)
            sim = torch.cosine_similarity(query_emb, doc_emb, dim=0)
            similarities.append(sim.item())
        
        best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
        return best_idx, similarities[best_idx], documents[best_idx]

# Usage example
similarity_engine = DocumentSimilarity()

documents = [
    "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
    "Deep learning uses neural networks with multiple layers to learn complex patterns.",
    "Natural language processing helps computers understand and generate human language.",
    "Computer vision enables machines to interpret and understand visual information.",
    "The weather today is sunny with a chance of rain in the afternoon."
]

query = "What is artificial intelligence and machine learning?"

best_idx, score, best_doc = similarity_engine.find_most_similar(query, documents)
print(f"Query: {query}")
print(f"Most similar document (score: {score:.3f}):")
print(f"  {best_doc}")

# Compute pairwise similarities
print("\nPairwise similarities:")
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        sim = similarity_engine.compute_similarity(documents[i], documents[j])
        print(f"Doc {i} vs Doc {j}: {sim:.3f}")
```

### Example 10: Multilingual Processing

```python
def multilingual_example():
    """Demonstrate multilingual capabilities"""
    
    # Byte mode handles any Unicode text
    config = PreciousConfig(mode="byte", d_model=256, n_layers=4)
    model = PreciousModel(config).to(device)
    
    # Multilingual test cases
    multilingual_texts = [
        "Hello, how are you?",           # English
        "Bonjour, comment allez-vous?",  # French
        "Hola, ¿cómo estás?",           # Spanish
        "こんにちは、元気ですか？",         # Japanese
        "你好，你好吗？",                # Chinese
        "Здравствуйте, как дела?",       # Russian
        "مرحبا، كيف حالك؟",             # Arabic
        "नमस्ते, आप कैसे हैं?"           # Hindi
    ]
    
    print("Processing multilingual texts:")
    
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i, text in enumerate(multilingual_texts):
            outputs = model([text])
            embedding = outputs["logits"].mean(dim=1)  # Pool over sequence
            embeddings.append(embedding)
            
            print(f"{i+1:2d}. {text[:30]:<30} -> Shape: {outputs['logits'].shape}")
    
    # Compute similarity matrix between languages
    print("\nCross-lingual similarity matrix:")
    embeddings_tensor = torch.cat(embeddings, dim=0)  # [num_texts, d_model]
    
    # Compute cosine similarities
    normalized_embs = torch.nn.functional.normalize(embeddings_tensor, dim=1)
    similarity_matrix = torch.mm(normalized_embs, normalized_embs.t())
    
    languages = ["EN", "FR", "ES", "JA", "ZH", "RU", "AR", "HI"]
    
    print("     ", " ".join(f"{lang:>6}" for lang in languages))
    for i, lang in enumerate(languages):
        similarities = [f"{similarity_matrix[i, j].item():6.3f}" for j in range(len(languages))]
        print(f"{lang}: {' '.join(similarities)}")

# Run multilingual example
multilingual_example()
```

### Example 11: Real-time Text Processing

```python
import threading
import queue
import time

class RealTimeProcessor:
    """Real-time text processing with Precious model"""
    
    def __init__(self, model_config=None, batch_size=4, max_wait_time=0.1):
        if model_config is None:
            model_config = PreciousConfig(mode="byte", d_model=256, n_layers=3)
        
        self.model = PreciousModel(model_config).to(device)
        self.model.eval()
        
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        
        # Thread-safe queues
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        
        # Processing thread
        self.processing_thread = None
        self.stop_processing = threading.Event()
    
    def start(self):
        """Start the processing thread"""
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop(self):
        """Stop the processing thread"""
        self.stop_processing.set()
        if self.processing_thread:
            self.processing_thread.join()
    
    def _processing_loop(self):
        """Main processing loop"""
        batch_buffer = []
        request_ids = []
        last_process_time = time.time()
        
        while not self.stop_processing.is_set():
            try:
                # Try to get new requests
                while (len(batch_buffer) < self.batch_size and 
                       (time.time() - last_process_time) < self.max_wait_time):
                    try:
                        request_id, text = self.input_queue.get(timeout=0.01)
                        batch_buffer.append(text)
                        request_ids.append(request_id)
                    except queue.Empty:
                        break
                
                # Process batch if we have requests
                if batch_buffer:
                    start_time = time.time()
                    
                    with torch.no_grad():
                        outputs = self.model(batch_buffer)
                        embeddings = outputs["logits"].mean(dim=1)  # Pool over sequence
                    
                    process_time = time.time() - start_time
                    
                    # Send results back
                    for i, request_id in enumerate(request_ids):
                        result = {
                            "embedding": embeddings[i].cpu().numpy(),
                            "processing_time": process_time,
                            "text_length": len(batch_buffer[i])
                        }
                        self.output_queue.put((request_id, result))
                    
                    # Reset batch
                    batch_buffer = []
                    request_ids = []
                    last_process_time = time.time()
                
            except Exception as e:
                print(f"Processing error: {e}")
                # Clear batch on error
                batch_buffer = []
                request_ids = []
    
    def process_text(self, text, request_id=None):
        """Submit text for processing"""
        if request_id is None:
            request_id = time.time()
        
        self.input_queue.put((request_id, text))
        return request_id
    
    def get_result(self, request_id, timeout=1.0):
        """Get processing result"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result_id, result = self.output_queue.get(timeout=0.1)
                if result_id == request_id:
                    return result
                else:
                    # Put back result for other requests
                    self.output_queue.put((result_id, result))
            except queue.Empty:
                continue
        
        return None  # Timeout

# Usage example
def demo_realtime_processing():
    """Demonstrate real-time processing"""
    processor = RealTimeProcessor(batch_size=2, max_wait_time=0.05)
    processor.start()
    
    # Simulate real-time requests
    texts = [
        "First real-time request",
        "Second request comes quickly",
        "Third request with different content",
        "Fourth request for batching demo",
        "Final request to complete the demo"
    ]
    
    print("Submitting real-time requests...")
    request_ids = []
    
    # Submit requests with small delays
    for i, text in enumerate(texts):
        request_id = processor.process_text(text)
        request_ids.append(request_id)
        print(f"Submitted request {i+1}: {text}")
        time.sleep(0.03)  # Small delay between requests
    
    # Collect results
    print("\nCollecting results...")
    for i, request_id in enumerate(request_ids):
        result = processor.get_result(request_id, timeout=2.0)
        if result:
            print(f"Request {i+1} result:")
            print(f"  Processing time: {result['processing_time']:.4f}s")
            print(f"  Text length: {result['text_length']} chars")
            print(f"  Embedding shape: {result['embedding'].shape}")
        else:
            print(f"Request {i+1}: Timeout!")
    
    processor.stop()
    print("Real-time processing demo completed.")

# Run the demo
demo_realtime_processing()
```

This completes the comprehensive examples document with:

1. **Getting Started**: Basic setup and first model
2. **Basic Usage**: Classification, language modeling, text generation
3. **Advanced Training**: Multi-task learning, curriculum learning
4. **Mode Comparison**: Performance comparison across T-FREE, CANINE, and byte modes
5. **Custom Training**: Advanced trainer with validation and checkpointing
6. **Performance Optimization**: Memory-efficient training techniques
7. **Real-world Applications**: Document similarity, multilingual processing, real-time processing

The examples progress from simple to complex, showing practical applications and optimization techniques for the Precious package.