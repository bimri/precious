# Changelog

All notable changes to the Precious package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-27

### Added
- Initial release of precious-nlp tokenizer-free NLP library
- Three tokenizer-free approaches:
  - **T-FREE**: Vocabulary-aware approach with character-level fallback
  - **CANINE**: Character-level processing with Unicode support and downsampling/upsampling
  - **Byte-level**: Direct byte-level text processing for universal text handling
- Complete transformer architecture with EVA attention mechanism
- Unified `PreciousModel` class supporting all three modes via configuration
- Comprehensive configuration system with `PreciousConfig` dataclass
- Automatic vocabulary building for T-FREE mode
- Character-level LSTM encoding for out-of-vocabulary words in T-FREE
- Bidirectional processing capabilities
- Support for variable-length sequences with automatic padding
- Device-agnostic design (CPU/CUDA support)

### Core Components
- `TFreeEncoder`: Complete implementation with vocabulary management
- `TFreeMLHead`: Multi-label prediction head for T-FREE models
- `CanineEmbedding`: Character-level embeddings with Unicode support
- `CanineDownUp`: Downsampling and upsampling for sequence length reduction
- `EVAAttention`: Enhanced vanilla attention with causal masking support
- `PositionalEncoding`: Sinusoidal positional embeddings up to 4096 tokens
- `TransformerBlock`: Standard transformer block with pre-layer normalization

### Testing & Quality Assurance
- Comprehensive test suite with 16 automated tests
- Integration tests for all three modes
- Performance benchmarking framework
- Memory efficiency testing
- Batch processing validation
- Device compatibility testing
- Error handling verification

### Documentation
- Complete API reference documentation (400+ lines)
- Extensive examples and tutorials (900+ lines)
- Implementation summary and technical overview
- Usage examples from basic to advanced use cases
- Performance optimization guidelines
- Multi-task learning examples
- Real-time processing examples
- Multilingual processing demonstrations

### Performance Features
- Memory-efficient training support
- Gradient checkpointing compatibility
- Mixed precision training support
- Real-time batch processing
- Configurable model architectures (128-512 dimensions, 2-12 layers)
- Automatic parameter counting and memory monitoring

### Development Tools
- Advanced trainer with checkpointing and validation
- Curriculum learning implementation
- Multi-task learning framework
- Performance benchmarking tools
- Document similarity computation
- Text generation utilities

### Package Infrastructure
- Modern Python packaging with setuptools and pyproject.toml
- GitHub Actions workflow for automated publishing
- Comprehensive dependency management
- MIT license
- Proper package structure with src/ layout
- Development and optional dependencies
- Code quality tools configuration (pytest, black, isort)

### Requirements
- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy >= 1.19.0

### Supported Platforms
- Linux, macOS, Windows
- CPU and CUDA GPU support
- Python 3.8, 3.9, 3.10, 3.11, 3.12

[Unreleased]: https://github.com/bimri/precious/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bimri/precious/releases/tag/v0.1.0