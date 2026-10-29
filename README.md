# 🛡️ Scam Intelligence Copilot
### ML-Powered Scam Detection with Local AI Cybersecurity Explanations

Scam Intelligence Copilot is an AI-powered cybersecurity application that analyzes suspicious SMS messages and helps users understand potential scam indicators. It combines a traditional machine learning classifier, rule-based threat analysis, a locally running Large Language Model (LLM), and Hybrid Retrieval-Augmented Generation (RAG) to provide understandable explanations and practical safety recommendations.

The application is built using Python and Streamlit, with Ollama powering local AI explanations.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Objectives](#-objectives)
4. [Key Features](#-key-features)
5. [Technology Stack](#-technology-stack)
6. [System Architecture](#-system-architecture)
7. [How the Application Works](#-how-the-application-works)
8. [Generative AI Concepts Used](#-generative-ai-concepts-used)
9. [Project Structure](#-project-structure)
10. [Installation and Setup](#-installation-and-setup)
11. [Running the Application](#-running-the-application)
12. [Testing and Evaluation](#-testing-and-evaluation)
13. [Evaluation Results](#-evaluation-results)
14. [Example Use Cases](#-example-use-cases)
15. [Limitations](#-limitations)
16. [Future Enhancements](#-future-enhancements)

---

## 📖 Project Overview

Digital communication has made it easier for scammers to impersonate banks, organizations, delivery services, and other trusted entities. Suspicious messages may use urgency, threats, malicious links, or requests for sensitive information to manipulate recipients.

Scam Intelligence Copilot helps users examine suspicious messages by combining machine learning with generative AI.

Unlike a basic spam classifier that only provides a prediction, the application aims to explain **why a message may be suspicious**, identify warning signs, retrieve relevant cybersecurity guidance, and recommend safer actions.

The project also explores how Large Language Models and Retrieval-Augmented Generation can be used to build a practical cybersecurity assistance application.

## 🎯 Problem Statement

Traditional spam classifiers generally focus on categorizing messages and may not explain the reasons behind their predictions. Users may struggle to recognize phishing attempts, OTP theft, fraudulent prize offers, and other social engineering techniques.

The objective is to develop an interactive application that combines ML-based message classification with AI-generated explanations and retrieved cybersecurity guidance, helping users make more informed decisions when they encounter suspicious messages.

## 🎯 Objectives

- Classify messages as spam-like or non-spam-like using a trained machine learning model.
- Identify potential threat indicators such as urgency, sensitive-information requests, and account threats.
- Generate understandable cybersecurity explanations using a local LLM.
- Retrieve relevant cybersecurity guidance using TF-IDF and semantic retrieval.
- Combine retrieval methods using Reciprocal Rank Fusion (RRF).
- Display relevant cybersecurity references and recommended safe actions.
- Evaluate the classifier and retrieval pipeline using appropriate metrics.
- Explore the practical application of Generative AI concepts in cybersecurity.

## ✨ Key Features

### 1. Machine Learning-Based Classification
Uses a trained scikit-learn classifier to estimate whether a message resembles spam.

### 2. Threat Indicator Analysis
Identifies potential warning signs, including:
- Urgency or threats.
- Requests for passwords, OTPs, or sensitive information.
- Suspicious message patterns.

### 3. Local AI Explanations
Uses Ollama with the `llama3.2:3b` model to generate explanations and safety recommendations locally.

### 4. Hybrid Retrieval-Augmented Generation
Combines TF-IDF-based retrieval and semantic retrieval to find potentially relevant cybersecurity guidance.

### 5. Cybersecurity Knowledge Base
Retrieves guidance about phishing, account threats, passwords, verification codes, prize scams, safe responses, reporting suspicious messages, and delivery-fee scams.

### 6. Source References
Displays retrieved document titles, source information, URLs, and RRF ranking scores.

### 7. Evaluation Dashboard
Displays the baseline classifier's accuracy, precision, recall, and F1-score.

### 8. Retrieval Evaluation
Evaluates TF-IDF, semantic, and hybrid retrieval using Recall@3 and Mean Reciprocal Rank (MRR).

---

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web application |
| scikit-learn | Machine learning, TF-IDF and evaluation |
| Logistic Regression | Baseline message classification |
| Ollama | Local LLM execution |
| Llama 3.2 3B | AI-generated explanations |
| Sentence Transformers | Semantic text embeddings |
| `all-MiniLM-L6-v2` | Semantic retrieval model |
| Reciprocal Rank Fusion | Combines retrieval rankings |
| JSON | Cybersecurity knowledge base and test cases |
| Joblib | Loading the trained classifier |

---

## 🏗️ System Architecture

The application uses multiple components that perform different tasks.

```text
                 USER
                   |
                   v
          Streamlit Web Interface
                   |
                   v
          Message Preprocessing
                   |
          +--------+---------+
          |                  |
          v                  v
    ML Classifier      Threat Analyzer
          |                  |
          v                  |
    Spam Prediction          |
    and Score                |
          |                  |
          +--------+---------+
                   |
                   v
            AI Explanation
                   |
          +--------+---------+
          |                  |
          v                  v
     TF-IDF Retrieval   Semantic Retrieval
          |                  |
          +--------+---------+
                   |
                   v
        Reciprocal Rank Fusion
                   |
                   v
        Retrieved Cybersecurity
               Guidance
                   |
                   v
          Local Ollama LLM
                   |
                   v
      Explanation and Safe Actions
                   |
                   v
       Results in Streamlit UI
```

The machine learning classifier estimates spam likelihood, while the threat analyzer identifies warning signs. The retrieval system provides supporting cybersecurity information to the LLM, which generates a natural-language explanation.

The classifier score and RRF ranking score represent different quantities and must not be interpreted as interchangeable confidence probabilities.

---

## ⚙️ How the Application Works

### Step 1: Message Input
The user enters a suspicious SMS or email message into the Streamlit interface.

### Step 2: ML Classification
The trained classifier analyzes the message and produces a spam-like prediction and an estimated spam score.

### Step 3: Threat Analysis
The threat analyzer checks for warning signs such as urgency, threats, and sensitive-information requests.

### Step 4: Knowledge Retrieval
The retrieval pipeline searches the cybersecurity knowledge base using two approaches:

- **TF-IDF retrieval:** Finds documents using lexical term relevance.
- **Semantic retrieval:** Finds documents using sentence embeddings and semantic similarity.

### Step 5: Hybrid Retrieval
Reciprocal Rank Fusion combines the rankings from the retrieval methods to produce a combined document ranking.

### Step 6: AI Explanation
The application sends the message and relevant retrieved context to the local LLM through Ollama.

### Step 7: Results Display
The application displays the prediction, threat indicators, explanation, recommended safe actions, retrieved guidance, and available evaluation metrics.

---

## 🧠 Generative AI Concepts Used

This project demonstrates concepts relevant to Generative AI and LLM applications.

### Large Language Models (LLMs)
The application uses Llama 3.2 3B through Ollama to generate natural-language cybersecurity explanations.

### Prompt Engineering
Instructions guide the LLM to summarize the message, explain potential warning signs, and suggest safe actions.

### Retrieval-Augmented Generation (RAG)
Retrieved cybersecurity guidance is supplied as context to support the generated explanation.

### Semantic Search
Sentence Transformers converts text into embeddings to help identify semantically related guidance.

### Hybrid Retrieval
TF-IDF and semantic rankings are combined using Reciprocal Rank Fusion to improve document retrieval.

### Responsible AI
The application communicates that its predictions are indicative, not proof of fraud. Users are encouraged to verify suspicious requests through trusted channels.

---

## 📂 Project Structure

The following is the intended structure based on the project's implemented components. Keep any additional files already present in your repository.

```text
ScamIntelligenceCopilot/
│
├── app.py
├── detector.py
├── threat_analyzer.py
├── llm_explainer.py
├── evaluation.py
├── rag_engine.py
├── semantic_rag.py
├── hybrid_rag.py
│
├── test_rag.py
├── test_semantic_rag.py
├── test_hybrid_rag.py
├── compare_retrieval.py
├── evaluate_rag.py
├── rag_test_cases.json
│
├── knowledge_base/
│   └── cybersecurity_guidance.json
│
├── models/
│   └── baseline_sms_classifier.joblib
│
├── data/
│   └── SMSSpamCollection
│
├── requirements.txt
├── README.md
└── .gitignore
```

**Note:** The exact files and paths depend on your local repository. Update this tree if your actual structure differs.

---

## 🚀 Installation and Setup

### Prerequisites

Install or prepare the following:

- Python compatible with the project's installed dependencies.
- Visual Studio Code or another Python development environment.
- Ollama.
- The `llama3.2:3b` model.
- The project's required data and trained model files.

### 1. Clone the Repository

If you are setting up the project from Git:

```bash
git clone YOUR_REPOSITORY_URL
cd ScamIntelligenceCopilot
```

Replace `YOUR_REPOSITORY_URL` with your actual repository URL.

If you already have the project locally, open its existing folder instead.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Install and Start Ollama

Install Ollama using its official website:

https://ollama.com/

Download the required model:

```bash
ollama pull llama3.2:3b
```

Confirm the model is available:

```bash
ollama list
```

Ollama must be running locally for the AI explanation feature to work.

### 5. Verify Required Files

Confirm that the trained classifier, cybersecurity knowledge base, and required dataset are available at the paths expected by the application.

The dataset may be needed for evaluation even when the trained classifier is already available.

---

## ▶️ Running the Application

From the project root, with the virtual environment activated, execute:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

Enter a suspicious message and inspect the prediction, warning indicators, AI explanation, retrieved guidance, and recommended actions.

### Running the Tests

Run the available test scripts individually:

```bash
python test_rag.py
python test_semantic_rag.py
python test_hybrid_rag.py
```

Evaluate retrieval quality:

```bash
python evaluate_rag.py
```

These commands test the existing retrieval components and evaluation pipeline. They do not, by themselves, establish that every possible application scenario works correctly.

---

## 📊 Testing and Evaluation

The project evaluates two distinct parts of the system.

### A. Machine Learning Evaluation

The baseline SMS classifier is evaluated using a held-out test split.

The reported metrics are:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted spam messages that are actually spam in the evaluation labels.
- **Recall:** Proportion of actual spam messages detected by the classifier.
- **F1-score:** Harmonic mean of precision and recall.

### B. RAG Evaluation

The retrieval pipeline uses manually labelled test cases and two ranking metrics.

- **Recall@3:** Measures whether relevant documents appear among the top three retrieved documents.
- **Mean Reciprocal Rank (MRR):** Measures how highly the first relevant document is ranked.

The retrieval evaluation compares TF-IDF, semantic retrieval, and Hybrid RAG.

These metrics assess different aspects of the system. Classifier accuracy does not measure the accuracy of the generated explanations or the quality of the retrieved sources.

---

## 📈 Evaluation Results

### Baseline Classifier

The application has reported the following results on its test split:

| Metric | Reported result |
|---|---:|
| Accuracy | 98.48% |
| Spam Precision | 94.00% |
| Spam Recall | 94.63% |
| Spam F1-score | 94.31% |
| Test messages | 1,115 |

These are previously observed evaluation results. Re-run the evaluation before reporting them as final results.

### Previously Observed Retrieval Baseline

| Retrieval method | Recall@3 | MRR |
|---|---:|---:|
| TF-IDF | 0.725 | 0.908 |
| Semantic | 0.704 | 0.842 |
| Hybrid | 0.787 | 0.900 |

These results were observed before the latest knowledge-base and test-case updates. They must not be presented as the final post-update results without re-running `python evaluate_rag.py`.

The previous baseline showed that Hybrid RAG had the highest Recall@3, while TF-IDF had the highest MRR.

---

## 🧪 Example Use Cases

### 1. Bank Phishing

**Example message:**

> URGENT: Your bank account will be blocked today. Click this link to verify your password immediately.

Potential warning signs include urgency, account threats, and a request to verify a password through a link.

### 2. OTP Scam

**Example message:**

> Share the OTP sent to your phone immediately to prevent your account from being suspended.

The application should identify the request for sensitive information and explain why sharing an unexpected verification code is dangerous.

### 3. Prize Scam

**Example message:**

> Congratulations! You won ₹50,000. Pay a ₹999 processing fee to receive your prize.

The explanation should discuss the risk of upfront fees associated with unexpected prize offers.

### 4. Delivery-Fee Scam

**Example message:**

> Your parcel is on hold. Pay a delivery release fee through this link to receive your package.

The application should retrieve relevant delivery-scam guidance when the appropriate knowledge-base entry is available.

### 5. Normal Message

**Example message:**

> Hi, our project meeting is scheduled for tomorrow at 11 AM. Please bring your notes.

This provides a useful ordinary-message test. The actual prediction should be recorded rather than assumed.

---

## ⚠️ Limitations

- The classifier was trained on an SMS spam dataset and is not a comprehensive scam detector.
- The model's spam score is an estimate, not proof that a message is fraudulent.
- Legitimate messages may be incorrectly classified, and malicious messages may be missed.
- Threat indicators are heuristic signals and do not independently establish fraud.
- Retrieval quality depends on the knowledge base, test labels, and retrieval configuration.
- Retrieved documents may be irrelevant or incomplete and should be checked.
- LLM-generated explanations can contain inaccuracies or unsupported statements.
- The application does not guarantee real-time verification of a sender, website, bank account, or transaction.
- Performance on the evaluation dataset may differ from performance on new messages and real-world scams.

Users should verify suspicious requests through independently trusted official channels and should never disclose passwords or OTPs in response to unexpected messages.

---

## 🔮 Future Enhancements

Potential improvements include:

- Expanding and diversifying the labelled training dataset.
- Adding more languages and regional scam patterns.
- Improving retrieval relevance and source verification.
- Evaluating the accuracy and factual grounding of generated explanations.
- Adding automated regression tests for the complete application.
- Exploring more advanced transformer-based classifiers.
- Introducing additional message-analysis features where appropriate.
- Improving the user interface, accessibility, and error handling.
- Deploying the application in a controlled environment with appropriate security safeguards.

These are proposed enhancements, not features claimed to be implemented in the current version.

---

## 🎓 Academic Relevance

Scam Intelligence Copilot demonstrates the integration of machine learning, local LLM inference, prompt engineering, semantic retrieval, and Retrieval-Augmented Generation in a cybersecurity use case.

It provides a practical way to explore how traditional classification and generative AI can complement one another: the classifier estimates spam likelihood, while the retrieval and language-generation pipeline helps explain potential warning signs and provide supporting guidance.

The project is intended for educational purposes and should not be treated as a substitute for professional cybersecurity investigation.

---

## 👩‍💻 Conclusion

Scam Intelligence Copilot combines ML-based message classification with local generative AI and Hybrid RAG to make suspicious-message analysis more understandable and actionable.

By providing an estimated prediction, identifying potential threat indicators, retrieving cybersecurity guidance, and generating safe-action recommendations, the project demonstrates a practical application of AI technologies to cybersecurity awareness.

**Remember: an AI prediction is a warning signal, not proof of fraud. Verify suspicious requests independently before taking action.**
