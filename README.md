# 🎬 Movie Genre Prediction

This project builds a **machine learning model** to predict a movie’s **genre** based on its textual information (title and description). It utilizes **TF-IDF vectorization** combined with a **Support Vector Machine (SVM)** classifier to classify movies into one of several genres.

---

## ✨ Features

* Parses raw training, test, and test-solution data.
* Converts movie plots into a numeric representation using **TF-IDF**.
* Trains a **Linear SVM (LinearSVC)** classifier with **balanced class weights** for robust performance.
* Evaluates the model using **classification reports** and **accuracy scores**.
* Predicts genres for the provided test set and writes results to `test_predictions.csv`.

---

## 🗂️ Dataset Format

Each data file contains movie information separated by `:::`:

* `train_data.txt`: `id ::: title ::: genre ::: description`
* `test_data.txt`: `id ::: title ::: description`
* `test_data_solution.txt`: `id ::: title ::: genre ::: description`

---

## 🧠 Workflow

1. **Load & parse data** using custom parser functions.
2. **Label encode** genres.
3. **Extract textual features** by concatenating the `title` and `description`.
4. **Train a pipeline** consisting of:

   * `TfidfVectorizer` with:

     * English stop words
     * Bigrams (`ngram_range=(1,2)`)
     * Vocabulary size capped (`max_features=20000`)
     * Filtering terms appearing in fewer than 3 documents
   * `LinearSVC` classifier with `class_weight='balanced'`
5. **Validate the model** on a stratified validation split and print performance.
6. **Test the model** on the test solution set to check its accuracy.
7. **Generate predictions** on the unseen test set and save them as `test_predictions.csv`.

---

## 🚀 Usage

1. Place your `train_data.txt`, `test_data.txt`, and `test_data_solution.txt` files in the project directory.
2. Install required dependencies:

   ```bash
   pip install pandas scikit-learn
   ```
3. Run the training and evaluation:

   ```bash
   python main.py
   ```
4. Check the `test_predictions.csv` file for the predicted genres.

---

## 📊 Example Output

* Training and validation performance is printed to the console.
* Final `classification_report` and `accuracy_score` on the test-solution file.
* Generated `test_predictions.csv` contains:

  * `id`
  * `title`
  * `predicted_genre`

---

## 🛠️ Dependencies

* Python 3.x
* `pandas`
* `scikit-learn`
