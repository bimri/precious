# Precious Documentation

Welcome to the Precious documentation! This directory contains comprehensive guides and references for using the Precious tokenizer-free NLP library.

## 📚 Available Documentation

### 📖 [API Reference](API_REFERENCE.md)
Complete API documentation covering all classes, methods, and configuration options.

**Contents:**
- `PreciousModel` - Main model class
- `PreciousConfig` - Configuration management
- Core components (`TFreeEncoder`, `CanineEmbedding`, `EVAAttention`)
- Training and inference examples
- Performance optimization tips

### 📝 [Examples](EXAMPLES.md)
Practical usage examples from basic to advanced implementations.

**Contents:**
- Quick start guide
- Three tokenizer-free approaches (Byte, CANINE, T-FREE)
- Text classification tasks
- Custom model implementations
- Benchmarking and evaluation

## 🚀 Quick Links

- **Installation**: See main [README](../README.md#installation)
- **GitHub Repository**: https://github.com/bimri/precious
- **PyPI Package**: https://pypi.org/project/precious-nlp/
- **Issues & Support**: https://github.com/bimri/precious/issues

## 📦 Package Overview

Precious provides three tokenizer-free approaches:

1. **Byte-Level Processing** - Universal and memory efficient
2. **CANINE Approach** - Character-level with Unicode support  
3. **T-FREE Method** - Vocabulary-aware with character-level fallback

```python
import precious
from precious import PreciousModel, PreciousConfig

# Choose your approach
config = PreciousConfig(mode="byte", d_model=256)  # or "canine", "tfree"
model = PreciousModel(config)

# Process text directly
outputs = model(["Hello, tokenizer-free world!"])
```

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](../README.md#contributing) for more information.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.