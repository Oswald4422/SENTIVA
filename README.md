# SENTIVA — Multilingual Sentiment Analysis for Ghanaian Text

---

## Project Overview

SENTIVA is a web-based sentiment analysis application built specifically for Ghanaian social media text. It classifies short pieces of text into three sentiment categories — **Positive**, **Neutral**, and **Negative** — with support for English, Asante Twi, Pidgin English, and code-switched text (text that mixes two or more languages in a single sentence).

The application combines a machine learning backend with a clean, responsive chat-style web interface. Users can type or paste text directly, or upload a plain text file for bulk analysis. Results are returned with a confidence percentage and a visual breakdown of probabilities across all three sentiment classes.

Live site: [https://sentiva-w6pe.onrender.com](https://sentiva-w6pe.onrender.com)

---

## Problem Statement

Social media in Ghana is uniquely multilingual. Users regularly mix English with Asante Twi, Pidgin English, or switch between languages within a single post. Standard sentiment analysis tools are trained almost exclusively on formal English text and fail to correctly interpret Ghanaian expressions, local slang, and code-switched content.

As a result, researchers, businesses, and institutions that want to understand public opinion from Ghanaian social media have no reliable automated tool to do so. Manual review is slow, expensive, and cannot scale to the volume of content produced daily.

SENTIVA addresses this gap by building a sentiment classifier trained specifically on multilingual Ghanaian social media content, with preprocessing designed to handle local language patterns.

---

## Objectives

1. Build and label a dataset of Ghanaian social media text that covers multiple languages and language mixing.
2. Develop a text preprocessing pipeline that handles local expressions, spelling variations, and Ghanaian language transliterations.
3. Train and compare multiple classification models to identify the best-performing approach.
4. Use ensemble methods to push accuracy beyond what individual models can achieve.
5. Deploy a user-friendly web application that makes the tool accessible to non-technical users.

---

## Key Features

- **Three-class sentiment classification** — Positive, Neutral, and Negative.
- **Multilingual support** — English, Asante Twi, Pidgin English, and code-switched text.
- **Single text analysis** — Type or paste any text and receive instant results.
- **Bulk file analysis** — Upload a `.txt` file with up to 200 lines and receive an aggregate sentiment breakdown with per-line results.
- **Confidence scores** — Each result shows a percentage confidence and a probability bar for all three sentiment classes.
- **Chat-style interface** — A familiar, easy-to-use conversation layout that preserves session history.
- **Responsive design** — Works on both desktop and mobile browsers.

---

## System Architecture

The system is divided into three main layers: a frontend web interface, a backend API server, and a machine learning pipeline.

> Full diagram: [`Resources/architecture.drawio`](Resources/architecture.drawio)  
> Open with [draw.io](https://app.diagrams.net) (File > Open from > Device).

**How the layers connect:**

- The **frontend** — a single self-contained HTML file — runs in the browser. The user enters text or uploads a file, and the interface sends that input to the backend as an HTTP request.
- The **backend** is a FastAPI server running in Python. It receives the request, passes the text through the machine learning pipeline, and returns a JSON response containing the sentiment label, confidence score, and a breakdown of probabilities.
- The **machine learning pipeline** preprocesses the text, converts it into numerical features using TF-IDF, and passes those features through a stacking ensemble of three models to produce a final prediction.
- On first run, the trained model is saved to disk. All subsequent server starts load the saved model instantly instead of retraining.

---

## Software Stack

| Component | Technology |
|---|---|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend framework | FastAPI (Python) |
| Web server | Uvicorn |
| Machine learning | scikit-learn |
| Data processing | pandas, numpy |
| Model persistence | joblib |
| UI/UX Design | Figma |
| Version control | Git, GitHub |
| Deployment | Render |

---

## User Workflow

> Full diagram: [`Resources/flowchart.drawio`](Resources/flowchart.drawio)  
> Open with [draw.io](https://app.diagrams.net) (File > Open from > Device).

1. The user opens SENTIVA in a web browser.
2. The welcome screen is displayed with three example prompts — one for each sentiment class.
3. The user either types text into the input field or clicks the upload button to select a `.txt` file.
4. The input is sent to the backend — single text goes to `/api/predict`, file uploads go to `/api/predict-file`.
5. The backend preprocesses the text, converts it to TF-IDF features, and runs it through the stacking ensemble.
6. A result card is displayed showing the sentiment label, confidence percentage, and a visual probability breakdown.
7. The user can continue submitting more text in the same session, with all results preserved in the chat history.

---

## Application Screenshots

**Welcome Screen**

The landing page when a user first opens the application. It displays the supported languages and three example prompts to help users get started immediately.

![Welcome Screen](<Resources/Screenshot 2026-06-22 010106.png>)

---

**Positive Sentiment Result**

Analysis of the text *"God bless you boss, you're doing great work!"* — classified as **Positive** with 81.7% confidence.

![Positive Sentiment Result](<Resources/Screenshot 2026-06-22 010125.png>)

---

**Multilingual Analysis — English and Asante Twi**

Two analyses in the same session. The English text is classified as Positive (81.7%) and the Twi text *"Mabr3 Ghana kraa"* (meaning "Ghana is exhausting") is classified as **Negative** (65.5%), demonstrating the application's ability to handle Ghanaian language input.

![Multilingual Analysis — English and Twi](<Resources/Screenshot 2026-06-22 010152.png>)

---

**All Three Sentiment Classes**

Three analyses in one session showing all three possible outcomes: Positive (English), Negative (Twi), and Neutral (*"Today be calm day"* in Pidgin English, 57.3%).

![All Three Sentiments](<Resources/Screenshot 2026-06-22 010216.png>)

---

**Extended Session — Pidgin English**

Continued analysis showing the Pidgin English phrase *"the sun dey give today waa"* classified as **Neutral** with 56.3% confidence, illustrating how the model handles idiomatic Pidgin expressions.

![Extended Session — Pidgin English](<Resources/Screenshot 2026-06-22 010456.png>)

---

## Challenges and Solutions

**No existing tools for Ghanaian multilingual text**  
Standard NLP libraries and pre-trained models do not recognize Twi or Pidgin English vocabulary. A custom preprocessing pipeline was built with a normalization map that standardizes common Ghanaian spellings and expressions — for example, both "kwasea" and "nkwasia" are mapped to "fool", and "paa" is mapped to "very".

**Code-switched text**  
Ghanaian social media often mixes languages within a single sentence. Rather than building separate models per language, the TF-IDF vectorizer was trained on the full mixed-language dataset, allowing it to learn vocabulary patterns across all languages simultaneously without requiring language detection.

**Class imbalance**  
The Neutral class made up nearly 40% of the dataset, while Positive and Negative were around 30% each. A stratified train/test split was used to ensure each class was proportionally represented in both training and evaluation, preventing the model from being biased toward the majority class.

**Model performance ceiling**  
Individual models (Naive Bayes, Logistic Regression, SVM) plateaued between 61% and 63% accuracy. A stacking ensemble was used to combine the outputs of all three base models through a meta-learner, which learned the best way to weight each model's vote and pushed accuracy to 64.6%.

**Slow startup time on deployment**  
The model took 30–60 seconds to train each time the server started, causing deployment platforms to time out before the server was ready to accept connections. The solution was to move model training to the build phase (using the build command rather than the start command) and save the trained model to disk with joblib. The server then loads the model instantly on startup.

---

## Future Improvements

- **Transformer-based models** — Fine-tuning a multilingual model such as mBERT or AfriBERTa on this dataset could significantly improve accuracy beyond the current 64.6%.
- **Larger dataset** — More labeled data, particularly for Twi and Pidgin, would improve model reliability and reduce uncertainty in edge cases.
- **Additional Ghanaian languages** — Extending support to Ga, Ewe, and Fante would make the tool more broadly useful across Ghana.
- **Live social media monitoring** — Integration with the Twitter/X API to analyze trending topics or keyword searches in real time.
- **Emotion detection** — Moving beyond positive/neutral/negative to detect specific emotions such as frustration, excitement, or concern.
- **User history and export** — Saving past analysis sessions and allowing users to export results as a CSV or PDF report.

---

## Lessons Learnt

**Preprocessing matters as much as the model itself.**  
For multilingual and low-resource text, the quality of text cleaning and normalization directly affects how well the model performs. Generic English preprocessing pipelines miss patterns that are specific to Ghanaian language use.

**Ensemble methods are worth the added complexity.**  
Combining three modest classifiers through stacking consistently outperformed any single model, with real improvements across precision, recall, and F1 score.

**Plan for deployment from the start.**  
The cold-start problem — where the model retrains on each server startup — was only discovered at the deployment stage. Designing the training and persistence strategy earlier would have saved significant debugging time.

**Interface quality shapes how a project is received.**  
A polished, responsive UI makes a machine learning project substantially more accessible and more compelling to present. The visual confidence breakdown helped users interpret results that might otherwise feel abstract.

**Honest evaluation builds credibility.**  
Reporting a real accuracy of 64.6% rather than inflating results with selective metrics gives the project integrity and sets clear, realistic expectations for anyone using or building on it.

---

## My Role

This project was a full-stack individual contribution within a group research context. My responsibilities covered every stage of the product, from design to deployment.

- **UI/UX Design** — Designed the complete interface in Figma before writing any code. This covered the welcome screen layout, chat interface structure, result card design, file upload flow, and responsive breakpoints for mobile.
- **Frontend Development** — Built the entire frontend as a self-contained HTML/CSS/JavaScript application, including the animated result cards, real-time confidence bars, file upload handling, toast notifications, and all responsive styling.
- **Machine Learning** — Developed the full ML pipeline: data loading and cleaning, text preprocessing with the Ghanaian language normalization map, TF-IDF feature extraction, and training all five models — Naive Bayes, Logistic Regression, SVM, Soft Voting Ensemble, and Stacking Ensemble.
- **Backend Development** — Built the FastAPI server including all API endpoints, CORS configuration, static file serving, request validation, and model persistence using joblib.
- **Integration** — Connected the frontend and backend, ensuring data flows correctly from user input through the REST API to the ML pipeline and back to the interface.
- **Deployment** — Configured and deployed the application on Render, including diagnosing and resolving the build-time model training issue to achieve fast server startup.
- **Research and Documentation** — Contributed to the research notebook, model evaluation with confusion matrices and confidence intervals, and the written project report.

---

## Project Structure

```
SENTIVA/
│
├── backend/
│   ├── main.py                  FastAPI server, API endpoints, and ML pipeline
│   └── requirements.txt         Python package dependencies
│
├── Dataset/
│   └── final_all_multilingual_tweets(1).csv
│                                2,468 labeled tweets used for training and evaluation
│
├── frontend/
│   └── index.html               Complete web interface (HTML, CSS, JavaScript in one file)
│
├── Resources/
│   ├── Group3ExamsReport.pdf    Full project thesis and research report
│   ├── architecture.drawio      System architecture diagram
│   ├── flowchart.drawio         User workflow flowchart
│   └── Screenshot *.png         Application screenshots (5 images)
│
├── Group3ExamsProject.ipynb     Research notebook — data exploration, model training, evaluation
├── start.bat                    Windows batch script to install dependencies and start the app
├── start.ps1                    Windows PowerShell script to install dependencies and start the app
├── .gitignore                   Files excluded from version control
└── README.md                    Project documentation (this file)
```

---

## Installation and Setup

**Requirements**
- Python 3.9 or higher
- pip

**Steps**

1. Clone the repository:

   ```bash
   git clone https://github.com/Oswald4422/SENTIVA.git
   cd SENTIVA
   ```

2. Install dependencies:

   ```bash
   pip install -r backend/requirements.txt
   ```

3. Start the server:

   ```bash
   cd backend
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

   On first run, the model will train and save to disk (approximately 30–60 seconds). All subsequent starts will load the saved model and be ready in under 5 seconds.

4. Open your browser and go to:

   ```
   http://localhost:8000
   ```

**Windows Shortcut**

Double-click `start.bat`, or right-click `start.ps1` and select "Run with PowerShell". Both scripts install dependencies and open the application automatically.

---

## Resources

| Resource | Link |
|---|---|
| Live Application | [https://sentiva-w6pe.onrender.com](https://sentiva-w6pe.onrender.com) |
| GitHub Repository | [https://github.com/Oswald4422/SENTIVA](https://github.com/Oswald4422/SENTIVA) |
| Project Thesis | [`Resources/Group3ExamsReport.pdf`](Resources/Group3ExamsReport.pdf) |
| System Architecture Diagram | [`Resources/architecture.drawio`](Resources/architecture.drawio) |
| User Workflow Diagram | [`Resources/flowchart.drawio`](Resources/flowchart.drawio) |
