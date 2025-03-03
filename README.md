# Question Paper Chatbot  

## Overview  
This project processes **question papers from images**, extracts text using **OCR (Tesseract)**, and generates **answers** using a language model. It also predicts **possible future questions** based on the extracted content.  

## Features  
- Extracts text from **scanned question papers**  
- Generates **detailed answers** using **FLAN-T5**  
- Predicts **possible upcoming questions**  
- **Preprocesses** extracted text to remove unwanted elements  
- Works on **Google Colab**  

## Installation & Setup  
### 1. Clone the Repository  
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```
### 2. Install Dependencies  
Run this in **Google Colab** or locally:  
```python
!pip install transformers torch torchvision torchaudio
!sudo apt install tesseract-ocr
!pip install pytesseract
```
### 3. Hugging Face Login (for FLAN-T5)  
```python
from huggingface_hub import login
login()
```
Get a token from [Hugging Face](https://huggingface.co/settings/tokens).  

## How It Works  
1. **Upload an image** of the question paper.  
2. **Extract questions** using OCR.  
3. **Generate answers** using **FLAN-T5**.  
4. **Predict future questions** based on patterns in the paper.  

## Example Output  
### Extracted Questions  
```
1. What is Machine Learning? Explain its types.  
2. Explain the concept of Bayesian Learning.  
3. What are the types of Activation Functions in Neural Networks?  
```
### Generated Answers  
```
1. Machine Learning is a field of AI that enables computers to learn from data. It has three types:  
   - Supervised Learning  
   - Unsupervised Learning  
   - Reinforcement Learning  

2. Bayesian Learning is a probabilistic approach where prior knowledge is updated with new data.  

3. Activation functions in Neural Networks include:  
   - ReLU  
   - Sigmoid  
   - Tanh  
   - Softmax  
```
### Predicted Future Questions  
```
1. Explain the difference between Deep Learning and Machine Learning.  
2. What is the role of Optimizers in Deep Learning?  
3. What is the Backpropagation Algorithm?  
```

