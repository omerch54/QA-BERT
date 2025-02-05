# **Question Answering (QA) System using Fine-tuned DistilBERT**  
**By: Omer Chaudhry**  

[Video Demonstration](https://drive.google.com/file/d/1Lj1lTMfEGBoUS92mIjuf3nHaozw5KQ9i/view?usp=drive_link)  

## **Project Overview**  
This project implements a **Question Answering (QA) system** using a fine-tuned **DistilBERT model**. The system is trained to:  

- **Predict Start and End Token Indices**: Identify the span of the answer in a given context.  
- **Classify Answer Type**: Determine whether an answer is present ("short answer") or absent ("no answer").  

The model is fine-tuned on a subset of the **Natural Questions dataset** and evaluated using **Precision, Recall, and F1-Score** to measure performance.  

---

## **Features**  
### **Data Processing**  
- Tokenization of question-context pairs using the `[CLS] Question [SEP] Context [SEP]` format.  
- Mapping of character-level answer spans to token indices.  
- Optional downsampling of null (no-answer) examples for better class balance.  

### **Model Architecture**  
- Uses **DistilBERT** as the backbone.  
- Fine-tuned with classifiers for:  
  - **Start and End token predictions** (identifying answer span).  
  - **Answer type classification** (short answer or no answer).  

### **Evaluation**  
- Joint loss computation for start, end, and type logits.  
- Metrics computation (Precision, Recall, and F1-score) on the validation set.  

---

## **Setup Instructions**  

### **1. Install Dependencies**  
Run the following command to install required libraries:  
```bash
pip install datasets==3.1.0 transformers torch evaluate tqdm
```

### **2. Prepare the Dataset**  
Upload your training and validation JSON files to Google Drive and then mount Google Drive in Colab:  
```python
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
```

### **3. Update Dataset Paths**  
Modify dataset paths to load your data:  
```python
data_files = {
    "train": "/content/drive/My Drive/Colab Notebooks/Copy of all_train.json",
    "dev": "/content/drive/My Drive/Colab Notebooks/Copy of all_dev.json"
}
dataset = load_dataset('json', data_files=data_files)
```

---

## **Key Components**  

### **Data Preprocessing**  
The `QADataset` class:  
- Tokenizes input **questions** and **contexts** using `DistilBertTokenizerFast`.  
- Maps character-level **answer spans** to token indices via `offset_mapping`.  

### **Model Architecture**  
The `BERTForQA` class:  
- Based on **DistilBERT** from HuggingFace's **Transformers** library.  
- Includes classifiers for:  
  - **Start and End token logits** (for answer span detection).  
  - **Type logits** (for answer classification: short answer or no answer).  

### **Training and Validation**  
The training pipeline includes:  
1. **train_loop**  
   - Trains the model for one epoch on the dataset.  
   - Computes training and validation loss.  
2. **eval_loop**  
   - Evaluates the model on the validation set.  
   - Computes Precision, Recall, and F1-score.  

### **Span Prediction**  
The `our_rank_spans` function:  
- Ranks spans based on start, end, and [CLS] token logits.  
- Returns the span with the highest score or predicts "no answer" if applicable.  

---

## **Usage**  

### **Train the Model**  
Run the training loop:  
```python
for epoch in range(num_epochs):
    avg_train_loss, avg_val_loss = train_loop(train_dataloader, validation_dataloader, model, optimizer, device)
    print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
```

### **Evaluate the Model**  
Evaluate the model on the validation set:  
```python
precision, recall, f1 = eval_loop(validation_dataloader, model, tokenizer, device)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
```

### **Make Predictions**  
Use the fine-tuned model to predict answers for custom question-context pairs:  
```python
def predict(question, context, model, tokenizer, device):
    inputs = tokenizer(
        question, context,
        max_length=512, truncation=True, padding="max_length",
        return_tensors="pt"
    ).to(device)

    model.eval()
    with torch.no_grad():
        start_logits, end_logits, type_logits = model(**inputs)
        cls_logit = type_logits[0, 0].item()
        predicted_span = our_rank_spans(start_logits[0], end_logits[0], cls_logit)

    return predicted_span

question = "What is the capital of France?"
context = "Paris is the capital of France."

predicted_span = predict(question, context, model, tokenizer, device)
print("Predicted Answer Span:", predicted_span)
```

---

## **Results**  
After training for 2 epochs, the following results are expected on the validation set:  
- **Precision:** ~0.67  
- **Recall:** ~0.67  
- **F1-Score:** ~0.65  
*(Results may vary based on dataset subset and hyperparameters.)*  

---

## **Hyperparameters**  
- **Learning Rate:** `3e-5`  
- **Batch Size:** `64`  
- **Epochs:** `2`  

---

## **Acknowledgments**  
This project is based on the paper **"A BERT Baseline for the Natural Questions"** and uses tools from the **HuggingFace Transformers** library.  
The dataset is a reduced version of the **Natural Questions dataset**.  

---

## **Future Enhancements**  
- Train on a larger dataset to improve generalization.  
- Fine-tune on additional QA datasets like **SQuAD** or **TriviaQA**.  
- Optimize the span-ranking function to improve answer selection.  
- Deploy the model via an API for real-world applications.  

---

## **Contributors**  
**Omer Chaudhry**  

