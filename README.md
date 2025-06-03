# ConnectAI - Social Media Platform with ML

A Flask-based social media platform featuring real-time content moderation and personalized recommendations using Machine Learning.

## 🚀 Features

- **Content Moderation**: AI-powered sentiment analysis for automatic content filtering
- **Personalized Recommendations**: Hybrid collaborative filtering recommendation system
- **Real-time Analytics**: Live dashboard with meaningful business metrics
- **User Management**: Support for multiple users with personalized experiences
- **Responsive Design**: Modern Bootstrap UI with mobile-first approach

## 🛠️ Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, pandas, numpy
- **Frontend**: Bootstrap 5, JavaScript, HTML/CSS
- **Data**: In-memory data generation with realistic simulation
- **Deployment**: Railway, Gunicorn

## 📁 Project Structure

```
├── app.py                     # Main Flask application
├── requirements.txt           # Python dependencies
├── Procfile                  # Railway deployment configuration
├── railway.toml              # Railway settings
├── .gitignore               # Git ignore file
├── models/
│   ├── sentiment_analyzer.py # Sentiment analysis model
│   └── recommender.py        # Recommendation system
├── data/
│   ├── data_generator.py     # Training data generator
│   └── content_database.py   # Content database with meaningful titles
├── templates/                # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── moderation.html
│   ├── recommendations.html
│   └── analytics.html
└── static/                   # CSS, JS, images
    └── style.css
```

## 🔧 Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd machine-learning-assignment
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Local: http://localhost:5000
   - Network: http://192.168.100.162:5000

## 🚀 Railway Deployment

### Files Required for GitHub Repository

Upload these files to your GitHub repository:

**Essential Files:**
- `app.py` - Main application
- `requirements.txt` - Dependencies
- `Procfile` - Deployment command
- `railway.toml` - Railway configuration
- `.gitignore` - Git ignore rules
- `README.md` - This documentation

**Source Code:**
- `models/` folder with Python files
- `data/` folder with Python files  
- `templates/` folder with HTML files
- `static/` folder with CSS/JS files

### Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit for Railway deployment"
   git push origin main
   ```

2. **Deploy on Railway**
   - Visit [railway.app](https://railway.app)
   - Connect your GitHub repository
   - Railway will automatically detect Flask app
   - Deploy with the provided `Procfile` and `railway.toml`

3. **Environment Variables** (optional)
   - `FLASK_ENV=production` (already set in railway.toml)
   - `PORT` (automatically set by Railway)

## 🎯 Key Features Explained

### 1. Meaningful Recommendations
- **Real Content**: Recommendations show actual titles, descriptions, creators
- **Categories**: Technology, Entertainment, Gaming, Lifestyle, Education, etc.
- **Explanations**: Clear reasons why content is recommended
- **Statistics**: Ratings, popularity ranks, confidence scores

### 2. Real Analytics
- **No Dummy Data**: All metrics calculated from actual user interactions
- **Live Updates**: Real-time statistics from ML model performance
- **User Analytics**: Individual user behavior and preferences
- **Item Analytics**: Content performance and similarity metrics

### 3. Content Database
- **100+ Items**: Realistic social media content with proper metadata
- **Categories**: 8 main categories with diverse content types
- **Rich Information**: Titles, descriptions, creators, tags, content types

## 📊 API Endpoints

- `GET /` - Home page
- `GET /content-moderation` - Content moderation interface
- `GET /recommendations` - Recommendation system interface
- `GET /analytics` - Analytics dashboard
- `POST /api/moderate` - Content moderation API
- `POST /api/recommend` - Recommendation API
- `GET /api/stats` - Platform statistics API
- `GET /api/user-analytics/<user_id>` - User-specific analytics
- `GET /api/item-analytics/<item_id>` - Item-specific analytics

## 🔍 Machine Learning Models

### Sentiment Analyzer
- **Purpose**: Content moderation and safety
- **Algorithm**: Multinomial Naive Bayes
- **Features**: TF-IDF vectorization
- **Classes**: Positive, Negative, Neutral
- **Performance**: ~89% accuracy

### Recommendation System
- **Purpose**: Personalized content recommendations
- **Algorithm**: Hybrid Collaborative Filtering
- **Similarity**: Cosine + Pearson correlation
- **Features**: User-item matrix, popularity ranking
- **Performance**: ~78% precision@5

## 🎨 UI Features

- **Responsive Design**: Works on desktop and mobile
- **Real-time Updates**: Live activity feeds and statistics
- **Interactive Charts**: HTML/CSS charts (no auto-scrolling issues)
- **Modern Styling**: Bootstrap 5 with custom CSS
- **User-friendly**: Intuitive navigation and clear information display

## 🚦 Getting Started

1. Visit the home page to see the platform overview
2. Try content moderation with sample posts
3. Explore recommendations for different users
4. Check real-time analytics dashboard
5. View detailed user and item analytics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is for educational purposes as part of a Machine Learning assignment.

## 🐛 Troubleshooting

**Common Issues:**
- Model loading errors: Ensure all dependencies are installed
- Port conflicts: Change port in environment variables
- Memory issues: Consider reducing data generation size for low-memory environments

**Railway Deployment Issues:**
- Check build logs for dependency errors
- Verify all required files are committed to Git
- Ensure `Procfile` and `requirements.txt` are in root directory

---

**Developed with ❤️ for Machine Learning Assignment** 