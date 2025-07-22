# 📚 Comparative Analysis of Graph Neural Network and Sequence Models for News Summarization

This project presents a **comparative study** of four different neural network architectures for text summarization:
- **Transformer**
- **BiLSTM**
- **GNN + Transformer**
- **GNN + BiLSTM**

The goal is to explore how **Graph Neural Networks (GNNs)** can enhance traditional sequence models by capturing document structure beyond simple sequential relationships. The project uses the **CNN/DailyMail dataset** and evaluates the models using **ROUGE metrics**.

---

## ✨ Key Features

- Implements **Transformer-based**, **BiLSTM**, and **GNN-enhanced hybrid** models.
- Builds **heterogeneous document graphs** with words and sentences as nodes.
- Uses **Graph Attention Networks (GATs)** for message passing between graph nodes.
- Integrates GNN representations with **BART Transformer** and BiLSTM.
- Includes **preprocessing pipeline**: cleaning, tokenization, graph construction.
- Evaluation using ROUGE-1, ROUGE-2, and ROUGE-L metrics.
- Reproducible notebooks: `GNN+Transformer.ipynb`, `GNN+BiLSTMs.ipynb`, `Transformer.ipynb`, `bilstm.ipynb`.

---

## ⚙️ Tech Stack

- **Language:** Python
- **Frameworks/Libraries:**
  - PyTorch
  - PyTorch Geometric
  - Transformers (Hugging Face)
  - NLTK
- **Models:** BART, BiLSTM, Graph Attention Networks
- **Dataset:** CNN/DailyMail

---
