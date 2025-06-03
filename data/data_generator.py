import pandas as pd
import numpy as np
import random

class DataGenerator:
    def __init__(self):
        self.random_state = 42
        np.random.seed(self.random_state)
        random.seed(self.random_state)
    
    def generate_sentiment_data(self, n_samples=2000):
        """Generate synthetic sentiment data"""
        
        # Positive examples
        positive_texts = [
            "I love this new social media platform! Great features! 😊",
            "Amazing user interface and smooth performance",
            "Best social media experience I've had in years!",
            "Having such a great time connecting with friends here!",
            "This platform is wonderful for sharing memories",
            "Love the design and functionality!",
            "Perfect way to stay connected with family ❤️",
            "Incredible features and great community support",
            "This app has changed how I interact online!",
            "Fantastic updates and new features!",
            "Great community and positive vibes 🎉",
            "Love how easy it is to use",
            "Amazing content discovery features",
            "Best recommendation system ever!",
            "Perfect for sharing photos and updates"
        ] * 40  # Repeat to get enough samples
        
        # Negative examples  
        negative_texts = [
            "This app is terrible and full of bugs",
            "I hate the new update, it's so confusing",
            "Worst social media platform ever created",
            "This is garbage and waste of time",
            "Terrible user experience and poor design",
            "I'm so frustrated with this app! 😠",
            "This platform is annoying and useless",
            "Horrible interface and confusing navigation",
            "I regret downloading this application",
            "This app is a complete disaster!",
            "The worst social media experience ever",
            "Buggy, slow, and unreliable",
            "I absolutely hate everything about this",
            "Complete waste of storage space",
            "Terrible customer service and support"
        ] * 40
        
        # Neutral examples
        neutral_texts = [
            "This is a social media platform",
            "I use this app sometimes",
            "It has some features that work okay",
            "The app exists and functions normally",
            "Some people like it, others don't",
            "It's an average social media application",
            "The features are standard for this type of app",
            "It works as expected for basic functionality",
            "The app is available for download",
            "Users can post and share content here",
            "It's like other social media platforms",
            "The interface is pretty standard",
            "Some features work better than others",
            "It does what it's supposed to do",
            "Regular social media functionality"
        ] * 40
        
        # Combine all texts and labels
        all_texts = positive_texts + negative_texts + neutral_texts
        all_labels = ([1] * len(positive_texts) + 
                     [0] * len(negative_texts) + 
                     [2] * len(neutral_texts))
        
        # Shuffle and take n_samples
        combined = list(zip(all_texts, all_labels))
        random.shuffle(combined)
        combined = combined[:n_samples]
        
        texts, labels = zip(*combined)
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': texts,
            'sentiment': labels
        })
        
        print(f"✅ Generated {len(df)} sentiment samples")
        print(f"📊 Distribution: {df['sentiment'].value_counts().to_dict()}")
        
        return df
    
    def generate_interaction_data(self, n_users=200, n_items=100, n_interactions=5000):
        """Generate synthetic user-item interaction data"""
        
        # Generate user-item interactions with realistic patterns
        user_ids = np.random.randint(1, n_users + 1, n_interactions)
        item_ids = np.random.randint(1, n_items + 1, n_interactions)
        
        # Create realistic rating distribution (higher ratings more common)
        ratings = np.random.choice([1, 2, 3, 4, 5], n_interactions, 
                                 p=[0.05, 0.10, 0.20, 0.35, 0.30])
        
        # Create DataFrame
        df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings
        })
        
        # Remove duplicates and keep only one rating per user-item pair
        df = df.drop_duplicates(subset=['user_id', 'item_id']).reset_index(drop=True)
        
        print(f"✅ Generated {len(df)} interaction samples")
        print(f"📊 Rating distribution: {df['rating'].value_counts().sort_index().to_dict()}")
        print(f"👥 Unique users: {df['user_id'].nunique()}")
        print(f"📱 Unique items: {df['item_id'].nunique()}")
        
        return df
    
    def generate_all_data(self):
        """Generate all required datasets"""
        print("🔄 Generating training data...")
        
        sentiment_data = self.generate_sentiment_data(n_samples=2000)
        interaction_data = self.generate_interaction_data(n_users=200, n_items=100, n_interactions=5000)
        
        return {
            'sentiment': sentiment_data,
            'interactions': interaction_data
        } 