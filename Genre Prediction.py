import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# ==============================
# 📥 Parse Functions
# ==============================
def parse_train_line(line):
    parts = line.strip().split(' ::: ')
    if len(parts) == 4:
        return {'id': parts[0], 'title': parts[1], 'genre': parts[2].lower(), 'description': parts[3]}

def parse_test_line(line):
    parts = line.strip().split(' ::: ')
    if len(parts) == 3:
        return {'id': parts[0], 'title': parts[1], 'description': parts[2]}

def parse_test_solution_line(line):
    parts = line.strip().split(' ::: ')
    if len(parts) == 4:
        return {'id': parts[0], 'title': parts[1], 'genre': parts[2].lower(), 'description': parts[3]}

def load_data(filepath, parser):
    with open(filepath, encoding='utf-8') as f:
        return pd.DataFrame([parser(line) for line in f if parser(line)])

# ==============================
# 🧠 Load Data
# ==============================
train_df = load_data('train_data.txt', parse_train_line)
test_df = load_data('test_data.txt', parse_test_line)
test_solution_df = load_data('test_data_solution.txt', parse_test_solution_line)

print(f'📂 Training samples: {len(train_df)}, Test samples: {len(test_df)}')

# ==============================
# 🔡 Encode Labels
# ==============================
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df['genre'])

# Combine title + description as input
X_train = train_df['title'] + ' ' + train_df['description']

# ==============================
# 🧠 Train Model (TF-IDF + SVM)
# ==============================
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        lowercase=True,
        max_features=20000,      # big vocab but manageable
        ngram_range=(1, 2),
        min_df=3
    )),
    ('clf', LinearSVC(
        C=1.0,
        class_weight='balanced',
        random_state=42,
        max_iter=5000
    ))
])

print('🚀 Training model...')
pipeline.fit(X_train, y_train)

# ==============================
# 🧪 Validate on a validation split
# ==============================
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, stratify=y_train, test_size=0.1, random_state=42
)
val_preds = pipeline.fit(X_train_split, y_train_split).predict(X_val)
print('📊 Validation performance:\n',
      classification_report(y_val, val_preds, target_names=label_encoder.classes_))

# Retrain on full training
pipeline.fit(X_train, y_train)

# ==============================
# 🔮 Predict on Test
# ==============================
X_test = test_df['title'] + ' ' + test_df['description']
test_preds = pipeline.predict(X_test)
test_df['predicted_genre'] = label_encoder.inverse_transform(test_preds)

# ==============================
# 🎯 Evaluate against Test Solution
# ==============================
y_true = label_encoder.transform(test_solution_df['genre'])
y_pred = pipeline.predict(test_solution_df['title'] + ' ' + test_solution_df['description'])

print('📉 Test set report:\n',
      classification_report(y_true, y_pred, target_names=label_encoder.classes_))
print('🎯 Test accuracy:', accuracy_score(y_true, y_pred))

# ==============================
# 💾 Save Predictions
# ==============================
test_df[['id', 'title', 'predicted_genre']].to_csv(
    'test_predictions.csv', index=False
)
