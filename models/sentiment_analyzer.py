import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
import os

class SentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        self.is_trained = False
        
        # Define sentiment keywords for better classification
        self.positive_words = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like',
            'happy', 'joy', 'pleased', 'satisfied', 'perfect', 'awesome', 'brilliant',
            'beautiful', 'nice', 'kind', 'friendly', 'helpful', 'best', 'better',
            'positive', 'success', 'win', 'winner', 'congratulations', 'thank', 'thanks',
            'appreciate', 'grateful', 'delighted', 'excited', 'thrilled'
        ]
        
        self.negative_words = [
            'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'dislike',
            'angry', 'mad', 'furious', 'disappointed', 'frustrated', 'upset', 'sad',
            'depressed', 'annoyed', 'irritated', 'worst', 'worse', 'fail', 'failure',
            'problem', 'issue', 'wrong', 'error', 'stupid', 'idiot', 'moron',
            'useless', 'worthless', 'pathetic', 'ridiculous', 'nonsense'
        ]
        
        self.neutral_words = [
            'okay', 'ok', 'fine', 'alright', 'normal', 'regular', 'standard',
            'average', 'medium', 'moderate', 'usual', 'typical', 'common',
            'general', 'basic', 'simple', 'plain', 'ordinary'
        ]
        
    def preprocess_text(self, text):
        """Simple text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep emojis and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s!?.,😊😢😡😍🎉👍👎❤️💻🌳☀️]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def create_balanced_sentiment_data(self, original_data):
        """Create balanced sentiment data with proper sentiment classification"""
        sentiment_data = []
        
        # Use the spam data but apply proper sentiment analysis
        for _, row in original_data.iterrows():
            text = str(row['text']) if 'text' in row else str(row['message'])
            original_label = row.get('original_label', 'unknown')
            
            # Apply rule-based sentiment classification
            processed_text = self.preprocess_text(text)
            words = processed_text.split()
            
            positive_score = sum(1 for word in words if word in self.positive_words)
            negative_score = sum(1 for word in words if word in self.negative_words)
            neutral_score = sum(1 for word in words if word in self.neutral_words)
            
            # Determine sentiment based on keyword analysis
            if positive_score > negative_score and positive_score > 0:
                sentiment_label = 1  # Positive
            elif negative_score > positive_score and negative_score > 0:
                sentiment_label = 0  # Negative
            else:
                # For neutral or unclear cases
                if original_label == 'spam':
                    sentiment_label = 0  # Spam is usually negative
                else:
                    sentiment_label = 2  # Neutral
            
            sentiment_data.append({
                'text': text,
                'sentiment': sentiment_label,
                'original_label': original_label
            })
        
        # Add some synthetic examples to ensure balance
        synthetic_examples = [
            # Positive examples
            ("I love this!", 1),
            ("This is amazing!", 1),
            ("You are wonderful!", 1),
            ("Great job everyone!", 1),
            ("Thank you so much!", 1),
            ("This makes me happy!", 1),
            ("Excellent work!", 1),
            ("You are a good person!", 1),
            ("Beautiful day today!", 1),
            ("I'm so excited!", 1),
            
            # Negative examples
            ("I hate this!", 0),
            ("This is terrible!", 0),
            ("You are annoying!", 0),
            ("This is disgusting!", 0),
            ("I'm so angry!", 0),
            ("Worst experience ever!", 0),
            ("This is stupid!", 0),
            ("I'm disappointed!", 0),
            ("This is awful!", 0),
            ("Complete failure!", 0),
            
            # Neutral examples
            ("The meeting is at 2pm", 2),
            ("Please send the report", 2),
            ("I'll be there soon", 2),
            ("The weather is okay", 2),
            ("Let me know when ready", 2),
            ("See you tomorrow", 2),
            ("The task is complete", 2),
            ("Information received", 2),
            ("Standard procedure", 2),
            ("Regular update", 2)
        ]
        
        for text, label in synthetic_examples:
            sentiment_data.append({
                'text': text,
                'sentiment': label,
                'original_label': 'synthetic'
            })
        
        return pd.DataFrame(sentiment_data)
    
    def train(self, data):
        """Train the sentiment analysis model with improved data"""
        print("🚀 Training Enhanced Sentiment Analysis Model...")
        
        # Create better sentiment data
        enhanced_data = self.create_balanced_sentiment_data(data)
        
        # Prepare data
        texts = enhanced_data['text'].apply(self.preprocess_text)
        labels = enhanced_data['sentiment']
        
        print(f"📊 Enhanced dataset: {len(enhanced_data)} samples")
        print(f"📊 Sentiment distribution: {labels.value_counts().to_dict()}")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create pipeline with TF-IDF and Logistic Regression
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000, 
                stop_words='english', 
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )),
            ('classifier', LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'  # Handle imbalanced classes
            ))
        ])
        
        # Train the model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Enhanced model trained successfully!")
        print(f"📊 Accuracy: {accuracy:.3f}")
        print(f"📈 Training samples: {len(X_train)}")
        print(f"🧪 Test samples: {len(X_test)}")
        
        # Print classification report
        class_names = ['Negative', 'Positive', 'Neutral']
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        self.is_trained = True
        self.save_model()
        
        return accuracy
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        if not self.is_trained and self.pipeline is None:
            raise ValueError("Model not trained or loaded!")
            
        processed_text = self.preprocess_text(text)
        prediction = self.pipeline.predict([processed_text])[0]
        probabilities = self.pipeline.predict_proba([processed_text])[0]
        
        # Map predictions to labels
        sentiment_labels = ['Negative', 'Positive', 'Neutral']
        sentiment = sentiment_labels[prediction]
        
        # Get confidence
        confidence = probabilities[prediction]
        
        return {
            'sentiment': sentiment,
            'confidence': float(confidence),
            'probabilities': {
                'negative': float(probabilities[0]),
                'positive': float(probabilities[1]),
                'neutral': float(probabilities[2])
            }
        }
    
    def moderate_content(self, text):
        """Perform content moderation with action recommendations"""
        result = self.predict_sentiment(text)
        
        # Calculate toxicity score (higher for negative sentiment)
        toxicity_score = result['probabilities']['negative']
        
        # Determine moderation action based on toxicity and confidence
        if toxicity_score > 0.7 and result['confidence'] > 0.6:
            action = "BLOCK"
            action_color = "danger"
        elif toxicity_score > 0.5 or result['confidence'] < 0.5:
            action = "REVIEW"
            action_color = "warning"
        else:
            action = "APPROVE"
            action_color = "success"
        
        return {
            'text': text,
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'toxicity_score': toxicity_score,
            'action': action,
            'action_color': action_color,
            'probabilities': result['probabilities']
        }
    
    def batch_moderate(self, texts):
        """Moderate multiple texts at once"""
        results = []
        for text in texts:
            result = self.moderate_content(text)
            results.append(result)
        return results
    
    def save_model(self):
        """Save the trained model"""
        if self.pipeline is not None:
            with open('models/sentiment_model.pkl', 'wb') as f:
                pickle.dump(self.pipeline, f)
            print("💾 Enhanced sentiment model saved!")
    
    def load_model(self):
        """Load a saved model"""
        try:
            with open('models/sentiment_model.pkl', 'rb') as f:
                self.pipeline = pickle.load(f)
            self.is_trained = True
            print("📂 Enhanced sentiment model loaded!")
        except FileNotFoundError:
            print("⚠️ No saved sentiment model found. Training new model...")
            return False
        return True
    
    def get_model_info(self):
        """Get information about the trained model"""
        if not self.is_trained:
            return {"status": "not_trained"}
            
        return {
            "status": "trained",
            "model_type": "Enhanced Logistic Regression + TF-IDF",
            "features": "5000 TF-IDF features with balanced classes",
            "classes": ["Negative", "Positive", "Neutral"],
            "training_time": "< 1 minute",
            "inference_time": "< 1ms per text",
            "enhancement": "Rule-based sentiment + synthetic examples for balance"
        } 