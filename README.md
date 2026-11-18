# 🧠 Artificial Intelligence & Machine Learning (AIML) Lab Practicals
### कृत्रिम बुद्धिमत्ता आणि मशीन लर्निंग (AIML) प्रयोग व लॅब मार्गदर्शिका

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/Repo-PrashilD15%2Faiml-black?logo=github)](https://github.com/PrashilD15/aiml)

Welcome to the **AIML Laboratory Practicals Repository**! This repository contains clean, modular, and fully tested Python implementations for college laboratory practicals (Practicals 2 to 9).

हे रिपॉझिटरी अभियांत्रिकी व कॉम्प्युटर सायन्सच्या **AIML प्रॅक्टिकल्स (2 ते 9)** साठी तयार करण्यात आले आहे. यामध्ये कोड कसा चालवायचा, कॉलेज लॅब पीसी आणि वैयक्तिक लॅपटॉपवर कसा एक्झिक्युट करायचा, आणि व्हायव्हा (Oral/Viva) मधील विचारले जाणारे महत्त्वाचे प्रश्न-उत्तरे इंग्रजी आणि मराठी दोन्ही भाषांमध्ये देण्यात आली आहेत.

---

## 📌 Dual-Environment Compatibility (दोन्ही वातावरणात चालण्यायोग्य)

- **College Lab PCs (कॉलेज लॅब पीसी):** Pre-configured with dataset path `/home/avcoe/`. (कॉलेजमधील लिनक्स पीसीवर कोड थेट चालेल.)
- **Personal Laptops / PCs (घरचा लॅपटॉप/कॉम्प्युटर):** Automatically falls back to the included `datasets/` directory in this repo. No extra setup required! (लॅपटॉपवर क्लोन केल्यावर `datasets/` फोल्डरमधून डेटा आपोआप लोड होईल.)

---

## 📑 Index of Practicals (प्रयोगांची अनुक्रमणिका)

| Practical | Topic / Title | Algorithm / Technique | Script Name |
| :---: | :--- | :--- | :--- |
| **02** | Route Planning / Graph Search | BFS & DFS | [`2.py`](2.py) / [`practical_02_bfs_dfs.py`](practical_02_bfs_dfs.py) |
| **03** | Expert System | Forward & Backward Chaining | [`3.py`](3.py) / [`practical_03_expert_system.py`](practical_03_expert_system.py) |
| **04** | Data Preprocessing & EDA | Cleaning, Summary & Train-Test Split | [`4.py`](4.py) / [`practical_04_data_preprocessing.py`](practical_04_data_preprocessing.py) |
| **05** | Supervised Classification | Logistic Regression & KNN | [`5.py`](5.py) / [`practical_05_classification_logistic_knn.py`](practical_05_classification_logistic_knn.py) |
| **06** | Regression Analysis | Linear & Polynomial Regression | [`6.py`](6.py) / [`practical_06_regression_linear_poly.py`](practical_06_regression_linear_poly.py) |
| **07** | Unsupervised Clustering | K-Means & Hierarchical Clustering | [`7.py`](7.py) / [`practical_07_clustering_kmeans_hierarchical.py`](practical_07_clustering_kmeans_hierarchical.py) |
| **08** | Dimensionality Reduction | PCA with SVM Classifier | [`8.py`](8.py) / [`practical_08_pca_dimensionality_reduction.py`](practical_08_pca_dimensionality_reduction.py) |
| **09** | Ensemble & Hyperparameter Tuning | GridSearchCV on SVM, RF & AdaBoost | [`9.py`](9.py) / [`practical_09_ensemble_hyperparameter_tuning.py`](practical_09_ensemble_hyperparameter_tuning.py) |

---

## 💻 How to Setup & Execute on Personal Laptop / PC
### लॅपटॉप किंवा कॉम्प्युटरवर कोड कसा रन करावा

### 1. Clone the Repository (प्रकल्प क्लोन करा):
Open Terminal (macOS/Linux) or PowerShell / Command Prompt (Windows):
```bash
git clone https://github.com/PrashilD15/aiml.git
cd aiml
```

### 2. Install Dependencies (लायब्ररी इन्स्टॉल करा):
```bash
pip install -r requirements.txt
```
*Or install manually:*
```bash
pip install pandas scikit-learn matplotlib seaborn numpy scipy
```

### 3. Run Any Practical (कोणतेही प्रॅक्टिकल रन करा):
You can run using either the short number filename (`2.py`) or the descriptive name:

```bash
# Practical 2
python 2.py
# किंवा
python practical_02_bfs_dfs.py

# Practical 3
python 3.py

# Practical 4
python 4.py

# Practical 5
python 5.py

# Practical 6
python 6.py

# Practical 7
python 7.py

# Practical 8
python 8.py

# Practical 9
python 9.py
```

---

## 🔬 Practical Details & Execution Commands (सविस्तर माहिती)

### Practical 2: Route Planning using BFS and DFS
- **English:** Explores graph traversal to find routes from start node `A` to goal node `F`. Breadth-First Search (BFS) explores level-by-level using a FIFO Queue and guarantees the shortest unweighted path. Depth-First Search (DFS) explores branch-by-branch using a LIFO Stack.
- **मराठी:** आलेखामधील (Graph) सुरुवातीच्या बिंदूपासून (`A`) अंतिम बिंदूपर्यंत (`F`) मार्ग शोधण्यासाठी BFS आणि DFS अल्गोरिदम वापरले आहेत. BFS कमीत कमी पायऱ्यांचा (Shortest Path) मार्ग शोधतो तर DFS एकाच शाखेमध्ये खोलवर शोध घेतो.
- **Run Command:**
  ```bash
  python 2.py
  ```

---

### Practical 3: Rule-Based Expert System (Forward & Backward Chaining)
- **English:** Implements a medical diagnosis expert system.
  - **Forward Chaining (Data-Driven):** Starts with known symptoms (`Fever`, `Cough`, `BodyPain`) and applies inference rules to deduce conclusions (`Flu` $\rightarrow$ `ViralInfection` $\rightarrow$ `Medicine`).
  - **Backward Chaining (Goal-Driven):** Starts with the hypothesis/goal (`Medicine`) and works backward to verify if premises are supported by known facts.
- **मराठी:** हे एक नियम-आधारित तज्ज्ञ प्रणाली (Expert System) आहे.
  - **फॉरवर्ड चेनिंग (डेटा-ड्रिव्हन):** आधीपासून माहिती असलेल्या लक्षणांवरून निष्कर्षापर्यंत पोहोचते.
  - **बॅकवर्ड चेनिंग (गोल-ड्रिव्हन):** अंतिम उद्दिष्टापासून (उदा. 'Medicine') सुरुवात करून पाठीमागे जाऊन तथ्ये तपासते.
- **Run Command:**
  ```bash
  python 3.py
  ```

---

### Practical 4: Data Preprocessing, EDA and Train-Test Split
- **Dataset:** `heart.csv` (Cleveland Heart Disease Dataset)
- **English:** Reads patient data, checks shapes, data types, summary statistics, fills missing values using median, removes duplicate rows, plots histograms and boxplots for `age` and `chol`, and splits into training and testing sets (80:20).
- **मराठी:** हृदयाच्या विकारासंबंधी डेटासेटवर डेटा क्लीनिंग, मिसिंग व्हॅल्यूज भरणे, डुप्लिकेट रेकॉर्ड्स काढणे, हिस्टोग्राम व बॉक्सप्लॉट द्वारे व्हिज्युअलायझेशन करणे आणि 80:20 प्रमाणात ट्रेन-टेस्ट स्प्लिट करणे.
- **Run Command:**
  ```bash
  python 4.py
  ```

---

### Practical 5: Classification using Logistic Regression & KNN
- **Dataset:** `Admission_Predict.csv` (Graduate Admissions Dataset)
- **English:** Encodes categorical columns, standardizes continuous features (`GRE Score`, `CGPA`) using `StandardScaler`, and trains both **Logistic Regression** and **K-Nearest Neighbors (KNN)**. Outputs accuracy, classification report (precision, recall, f1-score), and confusion matrices.
- **मराठी:** विद्यार्थ्यांच्या पदव्युत्तर प्रवेशाचा अंदाज लावण्यासाठी **लॉजिस्टीक रिग्रेशन** आणि **के-निअरेस्ट नेबर्स (KNN)** वर्गीकरण मॉडेल्स तयार करणे आणि त्यांची अचूकता तपासणे.
- **Run Command:**
  ```bash
  python 5.py
  ```

---

### Practical 6: Regression Analysis (Linear vs Polynomial)
- **Dataset:** `Housing.csv` (Housing Price Prediction)
- **English:** Handles missing values, encodes categorical features via One-Hot Encoding (`pd.get_dummies`), trains a standard **Linear Regression** model and a **Polynomial Regression (Degree 2)** model via a pipeline with `StandardScaler`. Compares performance using MSE, MAE, and $R^2$ Score.
- **मराठी:** घरांच्या किमतींचा अंदाज लावण्यासाठी **लिनिअर रिग्रेशन** आणि **पॉलिइनॉमिअल रिग्रेशन (Degree 2)** मॉडेल्सची तुलना MSE, MAE आणि $R^2$ गुणांकांच्या आधारे करणे.
- **Run Command:**
  ```bash
  python 6.py
  ```

---

### Practical 7: Clustering (K-Means & Hierarchical Agglomerative)
- **Dataset:** `Mall_Customers.csv` (Customer Mall Segmentation)
- **English:** Extracts `Age`, `Annual Income`, and `Spending Score`. Employs the **Elbow Method** to calculate Within-Cluster Sum of Squares (WCSS/inertia) to find optimal clusters ($K=5$). Displays 3D scatter plots of clusters and builds a **Dendrogram** with Ward's linkage for Hierarchical Agglomerative Clustering.
- **मराठी:** ग्राहकांच्या वय, उत्पन्न आणि खर्च करण्याच्या सवयींवरून **के-मीन्स (K-Means)** आणि **हायराक्रिकल (Dendrogram)** क्लस्टरिंग करणे.
- **Run Command:**
  ```bash
  python 7.py
  ```

---

### Practical 8: Dimensionality Reduction using PCA with SVM
- **Dataset:** `wdbc.csv` (Wisconsin Diagnostic Breast Cancer)
- **English:** Standardizes 30 diagnostic features, measures baseline SVM classification accuracy. Projects data to 2D for visualization via **PCA (Principal Component Analysis)**, and then reduces dimensions to preserve 95% of total variance. Re-trains SVM and demonstrates reduced feature count with maintained/improved accuracy.
- **मराठी:** 30 वैशिष्ट्यांची (features) संख्या **प्रिन्सिपल कॉम्पोनंट ॲनालिसिस (PCA)** द्वारे कमी करून 95% माहिती राखून SVM क्लासिफायरची अचूकता तपासणे.
- **Run Command:**
  ```bash
  python 8.py
  ```

---

### Practical 9: Model Optimization, Ensemble Learning & GridSearchCV
- **Dataset:** `breast_cancer.csv`
- **English:** Compares single classifier (**SVM**) against ensemble methods (**Random Forest** and **AdaBoost**). Optimizes hyperparameters using **GridSearchCV with 5-fold cross-validation** to identify best parameters and maximize Accuracy, Precision, Recall, and F1-score.
- **मराठी:** **SVM**, **रँडम फॉरेस्ट (Random Forest)**, आणि **ॲडाबूस्ट (AdaBoost)** मॉडेल्सचे **GridSearchCV** द्वारे हायपरपॅरामीटर ट्युनिंग करून सर्वोत्तम पॅरामीटर्स शोधणे.
- **Run Command:**
  ```bash
  python 9.py
  ```

---

## 🎯 Comprehensive Oral / Viva Voce Questions & Answers
### परीक्षेसाठी तोंडी (Oral / Viva) प्रश्न आणि उत्तरे

---

### 🔹 Practical 2: BFS and DFS
**Q1. What is the fundamental difference between BFS and DFS?**
- **English Answer:** BFS (Breadth-First Search) explores the search tree level by level using a **Queue (FIFO)** and guarantees finding the shortest path in an unweighted graph. DFS (Depth-First Search) explores deep along each branch before backtracking using a **Stack (LIFO)**.
- **मराठी उत्तर:** BFS मध्ये लेव्हलनुसार एकामागून एक नोड्स तपासले जातात (Queue द्वारे) आणि हा कमीत कमी अंतराचा मार्ग देतो. DFS मध्ये एका शाखेत खोलवर जाऊन तपासणी केली जाते (Stack द्वारे).

**Q2. What are the time and space complexities?**
- **BFS:** Time = $\mathcal{O}(V + E)$, Space = $\mathcal{O}(V)$ (requires storing all nodes of a level in the queue).
- **DFS:** Time = $\mathcal{O}(V + E)$, Space = $\mathcal{O}(d)$ (where $d$ is maximum depth).

---

### 🔹 Practical 3: Expert System & Chaining
**Q1. What is the difference between Forward Chaining and Backward Chaining?**
- **English Answer:**
  - **Forward Chaining:** Data-driven inference. It starts from known facts and applies rules to infer new facts until the goal is reached or no more rules can fire.
  - **Backward Chaining:** Goal-driven inference. It starts from a hypothesis/goal and searches backward for supporting facts.
- **मराठी उत्तर:** फॉरवर्ड चेनिंग आधीपासून माहीत असलेल्या तथ्यांवरून (Facts) पुढे जाऊन निष्कर्ष काढते (Data-driven). बॅकवर्ड चेनिंग आधी उद्दिष्ट (Goal) ठरवते आणि ते सिद्ध करण्यासाठी मागील पुरावे शोधते (Goal-driven).

**Q2. What is an Inference Engine?**
- **English Answer:** The brain of an expert system that applies logical rules to the knowledge base to deduce new information.
- **मराठी उत्तर:** इन्फरन्स इंजिन म्हणजे एक्सपर्ट सिस्टीमचा मेंदू, जो नॉलेज बेस मधील नियमांचा वापर करून निर्णय घेतो.

---

### 🔹 Practical 4: Data Preprocessing & EDA
**Q1. Why is Data Preprocessing essential before training ML models?**
- **English Answer:** Real-world datasets contain missing values, noise, outliers, and differing scales. Preprocessing prevents garbage-in, garbage-out, avoids model bias, and improves convergence speed and accuracy.
- **मराठी उत्तर:** कच्च्या डेटामध्ये गहाळ माहिती (Missing values) आणि त्रुटी असू शकतात. मॉडेलचे अचूक प्रशिक्षण होण्यासाठी डेटा साफ करणे अत्यंत आवश्यक असते.

**Q2. How do you decide between Mean and Median for filling missing values?**
- **English Answer:** Use **Mean** when data is normally distributed without extreme outliers. Use **Median** when data is skewed or contains significant outliers because median is resistant to extreme values.
- **मराठी उत्तर:** जर डेटामध्ये जास्त आउटलायर्स (Outliers) नसतील तर **Mean (सरासरी)** वापरतात. जर जास्त आउटलायर्स असतील तर **Median (मध्यक)** वापरतात कारण आउटलायर्सचा मीडियनवर परिणाम होत नाही.

---

### 🔹 Practical 5: Classification (Logistic Regression & KNN)
**Q1. Is Logistic Regression a regression algorithm or a classification algorithm?**
- **English Answer:** Despite its name, Logistic Regression is a **classification algorithm**. It uses the sigmoid (logistic) function to output probabilities between 0 and 1, which are then thresholded into classes.
- **मराठी उत्तर:** नावात रिग्रेशन असले तरी लॉजिस्टीक रिग्रेशन हा **वर्गीकरण (Classification)** अल्गोरिदम आहे. तो सिग्मॉइड फंक्शनच्या मदतीने शक्यता (Probability) मोजून वर्गीकरण करतो.

**Q2. Why is feature scaling (e.g., StandardScaler) mandatory for KNN?**
- **English Answer:** KNN calculates Euclidean distance between data points. If one feature has a range of 0-1000 and another 0-1, the larger feature will dominate the distance metric. Scaling ensures all features contribute equally.
- **मराठी उत्तर:** KNN मध्ये अंतराची (Distance) गणना केली जाते. जर एका कॉलमची व्हॅल्यू खूप मोठी असेल तर तो मॉडेलवर जास्त प्रभाव टाकतो. म्हणून सर्व फीचर्स समान प्रमाणात आणण्यासाठी स्केलिंग आवश्यक असते.

**Q3. What is a Confusion Matrix?**
- **English Answer:** A table used to evaluate classification performance containing True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

---

### 🔹 Practical 6: Regression Analysis
**Q1. What is the limitation of Linear Regression that Polynomial Regression solves?**
- **English Answer:** Linear Regression can only model straight-line (linear) relationships between dependent and independent variables. Polynomial Regression fits non-linear relationships by introducing higher-degree terms ($x^2, x^3$).
- **मराठी उत्तर:** लिनिअर रिग्रेशन फक्त सरळ रेषेतील संबंध दर्शवू शकते. जर डेटा वक्राकार किंवा नॉन-लिनिअर असेल, तर पॉलिइनॉमिअल रिग्रेशन अचूक अंदाज बांधतो.

**Q2. What does the $R^2$ score represent?**
- **English Answer:** The Coefficient of Determination ($R^2$) represents the proportion of variance in the dependent variable that is predictable from the independent variables (1.0 indicates perfect prediction).
- **मराठी उत्तर:** $R^2$ स्कोअर दर्शवतो की मॉडेलने डेटाचा किती टक्के फरक (Variance) अचूकपणे स्पष्ट केला आहे. १.० म्हणजे सर्वोत्तम मॉडेल.

---

### 🔹 Practical 7: Clustering (K-Means & Hierarchical)
**Q1. What is Unsupervised Learning?**
- **English Answer:** Learning from input data that does not have predefined labeled targets or ground truth; discovering inherent patterns or groupings.
- **मराठी उत्तर:** ज्या डेटामध्ये पूर्वीपासून लेबल्स किंवा उत्तरे दिलेली नसतात, त्यातील नमुने किंवा गट शोधणे म्हणजे अनसुपरव्हाइझ्ड लर्निंग.

**Q2. How do you determine the optimal number of clusters in K-Means?**
- **English Answer:** Using the **Elbow Method**: plot the Within-Cluster Sum of Squares (WCSS/inertia) against varying values of $K$. The point where the rate of decrease abruptly bends (like an elbow) is chosen as optimal $K$.
- **मराठी उत्तर:** एल्बो पद्धतीमध्ये (Elbow Method) वेगवेगळ्या क्लस्टर्ससाठी WCSS चे आलेख काढले जातात. ज्या ठिकाणी आलेखाचा उतार कोपऱ्यासारखा (Elbow) वाकतो, तो सर्वोत्तम $K$ मानला जातो.

**Q3. What is a Dendrogram?**
- **English Answer:** A tree diagram used in hierarchical clustering that illustrates the arrangement of clusters produced at each step of merging.
- **मराठी उत्तर:** डेंड्रोग्राम म्हणजे हायराक्रिकल क्लस्टरिंगमधील झाडासारखा आलेख (Tree Diagram) जो दर्शवतो की कोणकोणते बिंदू एकत्र आले आहेत.

---

### 🔹 Practical 8: PCA & Dimensionality Reduction
**Q1. What is the Curse of Dimensionality?**
- **English Answer:** As the number of features/dimensions grows, the volume of feature space increases exponentially, making the data sparse. This causes models to overfit, increases computational cost, and degrades distance metrics.
- **मराठी उत्तर:** जेव्हा वैशिष्ट्यांची (Features) संख्या खूप वाढते, तेव्हा मॉडेल हळू होते, जास्त मेमरी लागते आणि अचूकता कमी होते, याला कर्स ऑफ डायमेंशनालिटी म्हणतात.

**Q2. How does Principal Component Analysis (PCA) work?**
- **English Answer:** PCA projects high-dimensional data onto orthogonal axes (Principal Components) of maximum variance using eigenvalues and eigenvectors of the covariance matrix.
- **मराठी उत्तर:** PCA जास्त व्हेरियन्स (माहिती) असणाऱ्या मुख्य अक्षांवर डेटा प्रोजेक्ट करतो, ज्यामुळे अनावश्यक कॉलम कमी होतात आणि महत्त्वाची माहिती टिकून राहते.

---

### 🔹 Practical 9: Ensemble Learning & Hyperparameter Tuning
**Q1. What is Ensemble Learning? What is the difference between Bagging and Boosting?**
- **English Answer:**
  - **Ensemble Learning:** Combining multiple models (weak learners) to produce a stronger, more generalized predictor.
  - **Bagging (Bootstrap Aggregating, e.g., Random Forest):** Trains multiple models independently in parallel on random subsets of data and averages their votes to reduce variance.
  - **Boosting (e.g., AdaBoost):** Trains models sequentially, where each new model focuses on correcting the errors of previous models to reduce bias.
- **मराठी उत्तर:**
  - एकापेक्षा जास्त मॉडेल्स एकत्र करून अधिक अचूक मॉडेल बनवणे म्हणजे एन्सेम्बल लर्निंग.
  - **बॅगिंग (Random Forest):** अनेक मॉडेल्स स्वतंत्रपणे समांतर शिकवले जातात.
  - **बूस्टिंग (AdaBoost):** मॉडेल्स एकामागून एक शिकवले जातात आणि पुढील मॉडेल मागील मॉडेलच्या चुका सुधारते.

**Q2. What is GridSearchCV and why is Cross-Validation used?**
- **English Answer:** `GridSearchCV` exhaustively tests all specified combinations of hyperparameters for an estimator. Cross-Validation (e.g., 5-Fold) splits training data into $K$ partitions, ensuring hyperparameter selection generalizes well without overfitting to one split.
- **मराठी उत्तर:** ग्रिड सर्च सर्व दिलेल्या पॅरामीटर्सच्या जोड्या तपासून सर्वोत्तम जोडी निवडतो. क्रॉस-व्हॅलिडेशनमुळे मॉडेल एकाच डेटावर ओव्हरफिट होत नाही.

---

## 📁 Repository Structure (फायलींची रचना)

```
aiml/
│
├── 2.py / practical_02_bfs_dfs.py                          # Practical 2
├── 3.py / practical_03_expert_system.py                    # Practical 3
├── 4.py / practical_04_data_preprocessing.py               # Practical 4
├── 5.py / practical_05_classification_logistic_knn.py      # Practical 5
├── 6.py / practical_06_regression_linear_poly.py           # Practical 6
├── 7.py / practical_07_clustering_kmeans_hierarchical.py   # Practical 7
├── 8.py / practical_08_pca_dimensionality_reduction.py     # Practical 8
├── 9.py / practical_09_ensemble_hyperparameter_tuning.py   # Practical 9
│
├── datasets/                                               # Datasets for Laptop/PC
│   ├── heart.csv
│   ├── Admission_Predict.csv
│   ├── Housing.csv
│   ├── Mall_Customers.csv
│   ├── wdbc.csv
│   └── breast_cancer.csv
│
├── requirements.txt                                        # Dependencies
├── .gitignore
└── README.md                                               # Documentation (English & Marathi)
```

---

## 👨‍💻 Author & Contributions
- **Prashil Dalvi** ([@PrashilD15](https://github.com/PrashilD15))
- Amrutvahini College of Engineering, Sangamner (AVCOE)
