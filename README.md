# Clothing Review Recommendation Pipeline

Ever wondered if a handful of customer details and a few lines of text could predict whether someone will recommend a piece of clothing? That’s exactly what I tackled here—and spoiler: it works surprisingly well. This pipeline takes raw review text, customer age, feedback counts and product categories, then churns out a model that guesses “recommend” or “not recommend” with around 85% accuracy.

## Getting Started

Grab a copy of this repo and you’ll have everything you need to run the notebook on your own machine.

### Dependencies

Make sure you’ve got the following installed:

  Python 3.7 or higher  
  pandas
  numpy
  scikit-learn 
  spaCy
  seaborn
  matplotlib 
  Jupyter Notebook

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
jupyter notebook starter/starter.ipynb

## Testing

Time to see how our model performs on brand-new reviews. We feed the tuned Random Forest our hold-out test set and snap some key metrics—overall accuracy, sure, but also precision, recall and F1 for both “Yes, recommend” and “No, thanks.” And because numbers alone don’t always tell the whole story, we’ll wrap up with a confusion matrix so you can spot exactly where the model hit the mark—or missed it.

### Break Down Tests

- **accuracy_score**  
  This one’s straightforward: the percentage of reviews we labeled correctly out of the total.  
- **classification_report**  
  Here you get precision, recall and F1 broken out by class. In other words, how often we’re right when we say “recommend,” and how many “not recommend” slips through the cracks.  
- **confusion_matrix**  
  A quick heatmap that highlights which reviews got mixed up—the dark blocks show our strongest areas, and the lighter ones tell us where we can improve.

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Get predictions on our test set
y_pred = best_model.predict(X_test)

# Overall accuracy
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Per-class precision / recall / F1
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix heatmap for a clear visual of hits vs. misses
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Test Set")
plt.show()



## Project Instructions

This section should contain all the student deliverables for this project.

## Built With

* [Item1](www.item1.com) - Description of item
* [Item2](www.item2.com) - Description of item
* [Item3](www.item3.com) - Description of item

Include all items used to build project.

## License

[License](LICENSE.txt)
