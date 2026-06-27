import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import re
import nltk
from textblob import TextBlob

# ✅ FIX 1: Download all required NLTK data at the top
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)   # needed in newer NLTK versions

from nltk.corpus import stopwords        # import AFTER download


class SentimentAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.best_model = None
        self.best_name = None

        self.models = {
            'logistic_regression': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    min_df=2
                )),
                ('clf', LogisticRegression(max_iter=1000, C=1.0))
            ]),
            'naive_bayes': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2)
                )),
                ('clf', MultinomialNB(alpha=0.1))
            ]),
            'svm': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2)
                )),
                ('clf', LinearSVC(max_iter=1000))
            ]),
            'gradient_boosting': Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2)
                )),
                ('clf', GradientBoostingClassifier(n_estimators=100))
            ])
        }

    # ------------------------------------------------------------------ #
    def preprocess(self, text):
        """Clean and normalize raw text."""
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)         # remove HTML tags
        text = re.sub(r'http\S+', '', text)        # remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)   # keep letters only
        text = re.sub(r'\s+', ' ', text).strip()

        words = [
            w for w in text.split()
            if w not in self.stop_words and len(w) > 2
        ]
        return ' '.join(words)

    # ------------------------------------------------------------------ #
    def load_sample_data(self):
        """
        Built-in sample dataset — NO external download needed.
        1 000 rows (20 templates × 50 repetitions).
        """
        reviews = [
            'This product is amazing! Best purchase ever!',
            'Terrible quality. Broke after one day.',
            'Good value for money. Satisfied.',
            'Worst experience. Never buying again.',
            'Love it! Highly recommended to everyone.',
            'Average product. Nothing special.',
            'Outstanding quality and fast delivery!',
            'Very disappointed. Does not work.',
            'Perfect! Exactly what I needed.',
            'Complete waste of money. Do not buy!',
            'Great product, works perfectly.',
            'Poor quality, very disappointed.',
            'Excellent! Will buy again for sure.',
            'Not worth the price at all.',
            'Amazing quality, exceeded expectations!',
            'Product arrived damaged. Very unhappy.',
            'Five stars! Absolutely love this!',
            'Broke within a week. Terrible product.',
            'Good quality, fair price. Recommended.',
            'Horrible. Want my money back.',
        ]
        sentiments = [
            'positive', 'negative', 'positive', 'negative', 'positive',
            'neutral',  'positive', 'negative', 'positive', 'negative',
            'positive', 'negative', 'positive', 'negative', 'positive',
            'negative', 'positive', 'negative', 'positive', 'negative',
        ]
        return pd.DataFrame({
            'review':    reviews    * 50,
            'sentiment': sentiments * 50
        })

    # ------------------------------------------------------------------ #
    def train(self, df=None):
        """
        Train all models and keep the best one.

        ✅ FIX 2: 'results' dict is now created before the loop so
                  'return results' no longer raises NameError.
        """
        if df is None:
            df = self.load_sample_data()

        df['clean_review'] = df['review'].apply(self.preprocess)

        X = df['clean_review']
        y = df['sentiment']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("=" * 55)
        print("💬 SENTIMENT ANALYSIS - MODEL COMPARISON")
        print("=" * 55)

        best_accuracy = 0
        results = {}          # ✅ FIX 2: initialise before the loop

        for name, pipeline in self.models.items():
            pipeline.fit(X_train, y_train)
            y_pred    = pipeline.predict(X_test)
            accuracy  = accuracy_score(y_test, y_pred)
            results[name] = round(accuracy * 100, 2)   # ✅ save each score

            status = '🏆' if accuracy > best_accuracy else '  '
            print(f"{status} {name:25s} → {accuracy*100:.2f}%")

            if accuracy > best_accuracy:
                best_accuracy    = accuracy
                self.best_model  = pipeline
                self.best_name   = name

        print(f"\n🏆 BEST: {self.best_name} ({best_accuracy*100:.2f}%)")

        y_pred_best = self.best_model.predict(X_test)
        print(f"\n📋 Detailed Report ({self.best_name}):")
        print(classification_report(y_test, y_pred_best))

        return results        # ✅ FIX 2: always returns the dict

    # ------------------------------------------------------------------ #
    def predict(self, text):
        """
        Predict sentiment for a single review.

        ✅ FIX 3: guard against calling predict() before train().
        """
        if self.best_model is None:                       # ✅ FIX 3
            raise RuntimeError(
                "Model not trained yet. Call .train() first!"
            )

        clean      = self.preprocess(text)
        prediction = self.best_model.predict([clean])[0]

        # Confidence — not all models support predict_proba
        try:
            proba      = self.best_model.predict_proba([clean])[0]
            confidence = round(float(max(proba)) * 100, 2)
        except AttributeError:
            confidence = None          # LinearSVC has no predict_proba

        polarity = TextBlob(text).sentiment.polarity

        return {
            'text':       text,
            'sentiment':  prediction,
            'confidence': confidence,
            'polarity':   round(polarity, 3),
            'word_count': len(text.split())
        }

    # ------------------------------------------------------------------ #
    def analyze_batch(self, reviews):
        """Predict sentiment for a list of reviews and return a summary."""
        results    = [self.predict(r) for r in reviews]
        sentiments = [r['sentiment'] for r in results]

        return {
            'individual_results': results,
            'summary': {
                'total':        len(results),
                'positive':     sentiments.count('positive'),
                'negative':     sentiments.count('negative'),
                'neutral':      sentiments.count('neutral'),
                'avg_polarity': round(
                    np.mean([r['polarity'] for r in results]), 3
                )
            }
        }


# ========================== MAIN ============================
if __name__ == '__main__':

    analyzer = SentimentAnalyzer()

    # ── Train ──────────────────────────────────────────────
    results = analyzer.train()
    print("\n📊 All model accuracies (%):", results)

    # ── Single predictions ─────────────────────────────────
    test_reviews = [
        "This is the best phone I've ever bought!",
        "Terrible service. Never coming back.",
        "It's okay, nothing special.",
        "AMAZING!!! Love love love this product!",
        "Complete waste. Want refund immediately."
    ]

    emoji_map = {'positive': '😊', 'negative': '😠', 'neutral': '😐'}

    print("\n" + "=" * 55)
    print("🧪 TEST PREDICTIONS")
    print("=" * 55)

    for review in test_reviews:
        result = analyzer.predict(review)
        icon   = emoji_map.get(result['sentiment'], '❓')
        conf   = (f"  confidence: {result['confidence']}%"
                  if result['confidence'] else "")
        print(f"\n{icon} \"{review}\"")
        print(f"   → {result['sentiment'].upper()} "
              f"(polarity: {result['polarity']}){conf}")

    # ── Batch analysis ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("📦 BATCH ANALYSIS SUMMARY")
    print("=" * 55)

    batch = analyzer.analyze_batch(test_reviews)
    s = batch['summary']
    print(f"Total   : {s['total']}")
    print(f"Positive: {s['positive']} 😊")
    print(f"Negative: {s['negative']} 😠")
    print(f"Neutral : {s['neutral']}  😐")
    print(f"Avg polarity: {s['avg_polarity']}")