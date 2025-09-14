# **Flood-Related Question Classification**

This project involves fine-tuning the **GPT-2 model** for text classification, specifically to categorize questions related to flood scenarios into predefined types. Accurate question classification aids in disaster management by organizing textual data, enabling effective decision-making during emergencies.

---

## **Table of Contents**
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Installation and Setup](#installation-and-setup)
5. [Usage](#usage)
6. [Results](#results)
7. [Contributing](#contributing)
8. [Authors](#authors)

---

## **Introduction**

The purpose of this project is to develop a text classification model to categorize flood-related questions into distinct types (e.g., location, damage assessment, resource needs). This is critical for emergency response teams to process and prioritize queries efficiently during flood events. 

The project leverages **GPT-2**, a pre-trained transformer model, for sequence classification by fine-tuning it on a dataset specific to flood-related textual data.

---

## **Dataset**

- **Source**: Dataset from **FloodNet Challenge 2021 - Track 2**.
- **Format**: JSON file containing questions and their respective types.
- **Structure**:
  - `Question`: The text of the question.
  - `Question_Type`: The category/type of the question.

Example:
```json
{
  "1": {
    "Question": "How many buildings are flooded in this image?",
    "Question_Type": "Complex_Counting"
  },
  "2": {
    "Question": "Is the entire road non flooded?",
    "Question_Type": "Yes_No"
  }
}
```

---

## **Methodology**

1. **Data Preparation**:
   - The dataset is loaded from a JSON file and transformed into a pandas DataFrame.
   - Text (questions) and labels (question types) are extracted for processing.
   - Labels are converted into numerical categories using `pd.factorize`.

2. **Model and Tokenizer**:
   - GPT-2 tokenizer is used for text tokenization.
   - A GPT-2 model for sequence classification is configured with a number of labels corresponding to the unique question types.

3. **Training**:
   - Data is split into training and validation sets (70% train, 30% validate).
   - The model is fine-tuned using the Hugging Face `Trainer` API.
   - Training arguments include batch sizes, weight decay, warm-up steps, and logging settings.

4. **Model Saving**:
   - The fine-tuned model is saved for future use and deployment.

---

## **Installation and Setup**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AbdelkadirSellahi/Flood-Related-Question-Classification-Using-GPT-2.git
   cd Flood-Related-Question-Classification-Using-GPT-2
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   Required libraries include:
   - `transformers`
   - `torch`
   - `accelerate`
   - `pandas`
   - `sklearn`

3. **Download Dataset**:
   Place your dataset file in the `data/` directory.

4. **Run the Training Script**:
   ```bash
   python train.py
   ```

---

## **Usage**

### **Training**
To train the model with a custom dataset, edit the file path in the script:
```python
with open('/path/to/your/dataset.json', 'r') as file:
    data = json.load(file)
```
Then execute:
```bash
python train.py
```

### **Inference**
To use the trained model for inference:
1. Load the model:
   ```python
   from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
   model = GPT2ForSequenceClassification.from_pretrained('path/to/saved/model')
   tokenizer = GPT2Tokenizer.from_pretrained('path/to/saved/model')
   ```
2. Tokenize and predict:
   ```python
   inputs = tokenizer("Is the entire road non flooded?", return_tensors="pt", padding=True, truncation=True, max_length=512)
   outputs = model(**inputs)
   predicted_class = torch.argmax(outputs.logits, dim=-1).item()
   print(predicted_class)
   ```

---

## **Results**

- The model achieved **99% accuracy** on the validation set after fine-tuning.
- Evaluation metrics such as precision, recall, and F1-score are logged during training.

---

## **Authors**
- [**ABDELKADIR Sellahi**](https://github.com/AbdelkadirSellahi)
