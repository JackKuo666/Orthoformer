# Orthoformer

Orthoformer is a BERT-based pre-trained foundation model optimized for Orthologous Groups (OG) data. This repository contains the foundation model implementation and downstream task applications.

## Overview

Orthoformer is designed to process and understand orthologous group sequences, providing powerful feature representations for various bioinformatics downstream tasks. The model supports multiple positional encoding strategies (standard embeddings, ALiBi, and RoPE) and can handle sequences up to 2048 tokens.

## Project Structure

```
Orthoformer/
├── foundation_model/          # Foundation model implementation and usage
│   ├── README.md              # Detailed foundation model documentation
│   ├── requirements.txt       # Python dependencies
│   ├── orthoformer_model.py   # Model implementation (ALiBi & RoPE support)
│   ├── feature_extraction_example.py  # Feature extraction examples
│   ├── model/                 # Pre-trained model directory
│   └── datasets/              # Example datasets
├── Orthoformer_Operon/        # Operon prediction downstream task (to be implemented)
├── Orthoformer_Taxonomy/      # Taxonomy classification downstream task (to be implemented)
├── LICENSE                    # MIT License
└── README.md                  # This file
```

## Features

- **Pre-trained Foundation Model**: BERT-based architecture optimized for OG sequences
- **Multiple Positional Encodings**: Supports standard embeddings, ALiBi, and RoPE
- **Long Sequence Support**: Handles sequences up to 2048 tokens
- **GPU Acceleration**: Automatic GPU detection and utilization
- **Flexible Feature Extraction**: Multiple pooling strategies (CLS token, Mean pooling, Attention pooling)
- **Downstream Task Ready**: Foundation for various bioinformatics applications

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/Orthoformer.git
cd Orthoformer
```

2. Install dependencies:
```bash
cd foundation_model
pip install -r requirements.txt
```

### Model Download

Pre-trained models are available on Hugging Face:

**Model Repository**: https://huggingface.co/jackkuo/Orthoformer

For detailed download instructions, see [foundation_model/model/readme.md](foundation_model/model/readme.md)

Quick download:
```bash
pip install huggingface-hub
huggingface-cli download jackkuo/Orthoformer --local-dir ./foundation_model/model
```

### Basic Usage

See the [foundation_model/README.md](foundation_model/README.md) for detailed usage examples and API documentation.

Quick example:
```bash
cd foundation_model
python feature_extraction_example.py
```

## Foundation Model

The foundation model implementation and detailed documentation are located in the `foundation_model/` directory. Please refer to [foundation_model/README.md](foundation_model/README.md) for:

- Model architecture details
- Feature extraction methods
- Training and inference examples
- Model specifications

### Available Models

1. **model_3M_2048_v5**: Standard positional encoding
2. **model_3M_2048_v8**: ALiBi positional encoding (recommended for long sequences)

## Downstream Tasks

This repository includes the following downstream task implementations:

### Orthoformer_Operon
Operon prediction task using the Orthoformer foundation model. *(Implementation in progress)*

### Orthoformer_Taxonomy
Taxonomy classification task using the Orthoformer foundation model. *(Implementation in progress)*

## Requirements

- Python 3.12+
- CUDA-supported GPU (recommended, 8GB+ memory)
- PyTorch
- Transformers (Hugging Face)
- See [foundation_model/requirements.txt](foundation_model/requirements.txt) for full dependencies

## Citation

If you use Orthoformer in your research, please cite:

```bibtex
@article{orthoformer2025,
  title={Orthoformer: xxxxx},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions and issues, please open an issue on GitHub or contact the maintainers.

email: xxx

## Acknowledgments

We thank the open-source community for their valuable tools and libraries that made this project possible.
