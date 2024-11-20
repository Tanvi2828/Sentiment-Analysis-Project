# Sentiment Analysis on Financial Data

## Overview
This project implements sentiment analysis on a financial dataset using **Logistic Regression** and **TF-IDF** vectorization. The goal is to predict the sentiment of financial sentences (positive, negative, or neutral). 

## Dataset
The dataset contains two columns:
- **Sentence**: A text string representing a financial statement.
- **Sentiment**: A label indicating the sentiment of the sentence (positive, negative, neutral).

## Requirements
To run this project, ensure you have the following libraries installed:

- **pandas**
- **scikit-learn**
- **matplotlib**
- **seaborn**

You can install the required packages using:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Steps to Run the Project

1. **Data Loading**: Load the `data.csv` file containing financial sentences and their sentiment labels.
2. **Preprocessing**: Clean and preprocess the text data using **TF-IDF** vectorization.
3. **Model Training**: Train a **Logistic Regression** classifier using the processed data.
4. **Model Evaluation**: Evaluate the model using metrics such as accuracy, precision, recall, and F1-score. Display a **confusion matrix** to visualize the classification performance.
5. **Visualization**: Generate a heatmap of the confusion matrix to assess model performance.

## Usage
You can run the project by executing the following command:

```bash
python sentiment_analysis.py
```

This will:
- Train the model on the dataset.
- Output the evaluation metrics such as accuracy, precision, recall, and F1-score.
- Display the confusion matrix to visualize how well the model is performing.

## Results
The model achieved an **accuracy of 69.8%**, with the best performance on **neutral** and **positive** sentiments. The confusion matrix highlights the misclassification of **negative** sentiments, which could be improved with further model tuning or data balancing techniques.

## Improvements and Future Work
- **Data Balancing**: Oversampling or undersampling techniques could improve the classification of the **negative** sentiment.
- **Hyperparameter Tuning**: Fine-tuning the Logistic Regression model can help achieve better performance.
- **Model Comparison**: Exploring other algorithms like **SVM**, **Naive Bayes**, or ensemble methods (e.g., **Random Forest**).
- **Deep Learning**: Implementing advanced techniques like **LSTM** or **BERT** for better sentiment classification.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
