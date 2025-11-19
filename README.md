# 🚧 Fit-Track 2.0 — *Project In Progress*
AI-Powered Nutrition & Food Recognition System

Fit-Track 2.0 is an AI-driven food recognition and nutrition tracking system that detects food items from images, calculates calories and macro–micro nutrients, and provides personalized diet recommendations.  
Built using **YOLOv8, Flask, SQLite, and Ollama (Gemma-2B)**, the system delivers real-time dietary insights with high accuracy.

---

## 🚀 Features

- **Food Recognition (YOLOv8):** Detects multiple dishes from a single image.
- **Nutrition Estimation:** Calculates calories, macros, and micronutrients using curated datasets.
- **AI Nutrition Insights:** Uses Ollama (Gemma-2B) to generate health benefits and nutritional explanations.
- **Personalized Diet Recommendations:** Suggests meals based on user health goals.
- **Progress Tracking:** Logs meals and tracks calorie and nutrient trends.
- **Lightweight Backend:** Developed using Flask + SQLite for quick integration.

---

## 🧠 Tech Stack

- **AI Models:** YOLOv8, Ollama Gemma-2B  
- **Backend:** Flask (Python)  
- **Database:** SQLite  
- **ML Logic:** Decision Tree  
- **Datasets:** `dietdataset.csv`, `dietdataset1.csv`, `diet_recommandation.db`

---

## 📊 Model Performance

- **Top-1 Accuracy:** 96.58%  
- **Top-5 Accuracy:** 99.71%  
- **mAP (food recognition):** ~90%  
- **Inference Time:** ~274 ms per image  

---

## 🖼️ Example Output

- **Detected Items:** Idli, Dosa  
- **Calories:**  
  - Idli → 6 × 40 kcal  
  - Dosa → 2 × 106 kcal  
- **Confidence Scores:** 0.91, 0.89, 0.88…  
- **Final Output Includes:**  
  - Food detection  
  - Portion estimation  
  - Nutrition breakdown  
  - AI-generated health insights  

---

## 🔧 Installation (Coming Soon)

> Code will be uploaded once final testing & deployment are complete.

