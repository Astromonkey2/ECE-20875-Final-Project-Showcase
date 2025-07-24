# Data Security in Model Training: A Project Showcase

This project explores the vulnerabilities of machine learning models to data poisoning attacks and the effectiveness of denoising countermeasures. The analysis was conducted on the scikit-learn digits dataset to demonstrate how different model architectures react to adversarial noise.

This repository contains the source code for the project "Data Security in Model Training," completed for Purdue University's ECE 20875 course.

---

### Key Analysis Steps

* **Model Selection**: Implemented and evaluated three distinct classification models: Gaussian Naive Bayes, K-Nearest Neighbors (KNN), and a Multi-Layer Perceptron (MLP).
* **Data Poisoning Attack**: Simulated an adversarial attack by injecting significant Gaussian noise into the training data to degrade model performance.
* **Denoising with KernelPCA**: Applied Kernel Principal Component Analysis (KernelPCA) as a denoising technique to recover model performance after the attack.

### Key Findings

* The K-Nearest Neighbors (KNN) model, while the top performer on clean data (95.46% accuracy), was the most vulnerable to data poisoning, with its performance dropping by nearly 35%.
* The Gaussian Naive Bayes model was surprisingly robust against the noise, showing a slight improvement in performance on the poisoned data.
* Denoising with KernelPCA was most effective for the KNN model, restoring over 23% of its lost accuracy.

---

### How to Run This Project

**1. Prerequisites**
* Python 3.8+
* pip

**2. Installation**
Clone the repository and install the required packages:
```bash
git clone https://github.com/Astromonkey2/ECE-20875-Final-Project-Showcase.git
cd ECE-20875-Final-Project-Showcase
pip install -r requirements.txt
```

**3. Running the Analysis**
Execute the main script to run the full analysis pipeline:

```bash
python main.py
```

---

### Dataset
This project uses the Digits dataset included with scikit-learn, which consists of 8x8 pixel images of handwritten digits. The dataset is available under the BSD 3-Clause license.

---

### License
This project is licensed under the MIT License. See the LICENSE file for details.

---

### Academic Integrity & Attribution
This project was completed as coursework for Purdue University's ECE 20875. Please do not submit this work as your own for academic credit. The MIT license requires attribution if you use or adapt this code.

---

### Academic Integrity Note
This project is for demonstration and portfolio purposes only. Please respect academic integrity and do not submit this work, in part or in whole, as your own for academic credit.
