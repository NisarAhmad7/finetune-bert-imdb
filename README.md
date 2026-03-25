# BERT Sentiment Classifier (Finetuning)

This project demonstrates how to finetune a pretrained BERT model on the IMDB dataset for sentiment classification (positive / negative).

##  Features
- Uses pretrained `bert-base-uncased`
- Finetuned on IMDB dataset
- Simple and clean implementation using Hugging Face Transformers
- Ready for training and inference

##  Project Structure


.
├── train.py
├── requirements.txt
├── README.md
└── finetuned-bert/ # saved model after training


##  Installation

```bash
pip install -r requirements.txt
> Run Training
python train.py
> Model Details
Model: BERT (bert-base-uncased)
Task: Sentiment Classification
Labels:
0 → Negative
1 → Positive
> Output

After training, the model will be saved in:

./finetuned-bert
> Dataset
IMDB dataset (loaded via Hugging Face datasets)
> Tech Stack
Python
PyTorch
Transformers (Hugging Face)
Datasets