# 🔐 Artificial Familiarity Attack Detector (AFAD)

A hybrid AI system designed to detect **social engineering attacks based on artificial familiarity**, using a combination of Machine Learning, rule-based logic, and heuristic corrections.

---

## 🚀 Overview

Social engineering attacks often exploit **trust, urgency, and familiarity** rather than technical vulnerabilities.

This project detects such attacks by analyzing:

* Text patterns
* Financial intent
* Urgency signals
* Regional language usage (including Manglish)

The system outputs a **risk classification**:

* 🟢 Safe
* 🟡 Suspicious
* 🔴 Attack

---

## 🧠 System Architecture

```
Input Message
     ↓
TF-IDF Vectorizer
     ↓
Keyword Feature Extraction
     ↓
Logistic Regression Model
     ↓
Threshold Layer
     ↓
Rule-Based Safety Layer
     ↓
False Positive Reduction Layer
     ↓
Final Prediction
```

---

## ⚙️ Features

### ✅ 1. Machine Learning Core

* TF-IDF Vectorization (5000 features)
* Logistic Regression (`class_weight="balanced"`)
* Trained on a **custom curated dataset**

---

### ✅ 2. Keyword Feature Flags

Binary features:

* `has_money_term`
* `has_urgency`
* `has_payment_term`

These help the model understand **intent beyond text patterns**.

---

### ✅ 3. Rule-Based Safety Layer

Critical override:

```
IF money_term == 1 AND urgency == 1
→ Force Attack
```

Ensures important scams are **never missed**.

---

### ✅ 4. Threshold-Based Classification

| Probability | Label      |
| ----------- | ---------- |
| > 85%       | Attack     |
| 50% – 85%   | Suspicious |
| < 50%       | Safe       |

---

### ✅ 5. False Positive Reduction Layer

If no scam signals are present:

* Reduce risk score
* Prevent normal messages from being flagged

---

## 📊 Dataset

Final dataset:

* Total samples: **7820**
* Safe: **5000**
* Attack: **2820**

### Enhancements:

* Indian financial terms (lakh, crore, etc.)
* Payment platforms (GPay, UPI, Paytm, PhonePe)
* Manglish variations
* Diverse scam contexts

---

## 🧪 Example Predictions

| Input                  | Output    |
| ---------------------- | --------- |
| send 1 lakh now        | 🔴 Attack |
| gpay me 20k            | 🔴 Attack |
| bro money venam urgent | 🔴 Attack |
| meeting tomorrow       | 🟢 Safe   |
| hello how are you      | 🟢 Safe   |

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
cd AFAD_Project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Launch the Dashboard
The main interface is built with Streamlit. Run it using:
```bash
streamlit run app.py
```

### (Optional) Retrain the Model
If you wish to retrain the hybrid text model using the current dataset:
```bash
python models/train_model.py
```

---

## 💻 Programmatic Usage

You can also use the detection logic directly in your Python scripts:

```python
from scripts.utils_ui import predict_text

# Analyze a message
label, risk = predict_text("send 1 lakh now")

print(f"Classification: {label}")  # Output: Attack
print(f"Risk Score: {risk:.2f}%")
```

---

## 📁 Project Structure

```
AFAD_Project/
│
├── dataset/
├── models/
├── features/
├── scripts/
│   └── utils_ui.py
├── train_model.py
├── train_text_model.py
└── README.md
```

---

## 📈 Performance

* Accuracy: ~99%
* Attack Precision: ~99%
* Attack Recall: ~98%

> Note: Metrics are based on a curated dataset and may vary in real-world scenarios.

---

## ⚠️ Limitations

* Probability scores are adjusted using heuristics (not fully calibrated)
* Rule-based logic may require tuning for different regions
* Limited to text-based detection (voice/video planned)

---

## 🔮 Future Improvements

* Probability calibration (Platt scaling)
* Expanded multilingual support
* Real-time deployment (API / Web app)
* Integration with email/chat systems

---

## 👨💻 Author

Developed as part of an AI/ML project focused on **cybersecurity and social engineering detection**.

---

## 📜 License

This project is for educational purposes.
