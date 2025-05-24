# Clothing Review Recommendation Pipeline

Ever wondered if a handful of customer details and a few lines of text could predict whether someone will recommend a piece of clothing? That’s exactly what I tackled here—and spoiler: it works surprisingly well. This pipeline takes raw review text, customer age, feedback counts and product categories, then churns out a model that guesses “recommend” or “not recommend” with around 85% accuracy.

## Getting Started

Grab a copy of this repo and you’ll have everything you need to run the notebook on your own machine.

### Dependencies

matplotlib
spacy
pandas
numpy
scikit-learn
seaborn

### Installation

Just a few quick commands and you’ll be up and running:

-Clone this repo
-git clone https://github.com/MatthewJay412/dsnd-pipelines-project.git
-cd dsnd-pipelines-project

Install everything
pip install -r requirements.txt

Grab the spaCy English model
python -m spacy download en_core_web_sm

Launch the notebook
jupyter notebook starter.ipynb

## Testing

Time to see how our model performs on brand-new reviews. We feed the tuned Random Forest our hold-out test set and snap some key metrics—overall accuracy, sure, but also precision, recall and F1 for both “Yes, recommend” and “No, thanks.” And because numbers alone don’t always tell the whole story, we’ll wrap up with a confusion matrix so you can spot exactly where the model hit the mark—or missed it.

### Break Down Tests

- **accuracy_score**  
  This one’s straightforward: the percentage of reviews we labeled correctly out of the total.  
- **classification_report**  
  Here you get precision, recall and F1 broken out by class. In other words, how often we’re right when we say “recommend,” and how many “not recommend” slips through the cracks.  
- **confusion_matrix**  
  A quick heatmap that highlights which reviews got mixed up—the dark blocks show our strongest areas, and the lighter ones tell us where we can improve.

This is the function I used to calculate the metrics.
```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```
# Get predictions on our test set
y_pred = best_model.predict(X_test)

# Overall accuracy
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Per-class precision, recall, F1
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix heatmap for a visual of hits vs. misses
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Test Set")
plt.show()

## Project Instructions

This notebook lays out everything you need, from raw data to a tuned model. 
Here’s the rundown

**Data Loading & Inspection**  
  We kick off by loading the reviews CSV, checking data types, spotting any missing values, and glancing at basic stats.

**Exploratory Visualizations**  
  Next up, we dive into chart. Recommendation counts and proportions, numeric feature distributions, top-10 categories, plus a donut chart and a 2×2 grid of for visuals. Because one plot is rarely enough.

**Feature Engineering**  
 We whip up two quick features, review length and exclamation mark count. And we give our text a clean up using spaCy, lemmatizing and filtering out the noise.

**Preprocessing Pipeline**  
  Numeric fields get median imputation and scaling. Categorical bits go through a “fill-missing-and-one-hot” routine. Text is first cleaned with our custom TextCleaner and then vectorized via TF-IDF. After that, all numeric, categorical, and text transformations are combined with a ColumnTransformer into one cohesive Pipeline.

**Hyperparameter Tuning**  
  We put our Random Forest under the HalvingRandomSearchCV microscope, trying different tree counts and depths with 5-fold cross-validation to find the sweet spot.

**Model Training**  
  Once the best settings emerge, we train the final pipeline on our full training split.

**Model Evaluation**  
  At the end, we let the model loose on hold-out data. You get overall accuracy plus precision, recall, and F1 for both “Recommended” and “Not Recommended.” And of course, a confusion matrix heatmap to show where we nailed it (and where we can still improve).

## Built With

* [Item1](www.item1.com) - Description of item
* [Item2](www.item2.com) - Description of item
* [Item3](www.item3.com) - Description of item

Include all items used to build project.

## License

[License](LICENSE.txt)
