from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Mock models for testing
models = {
    'tfidf_logistic': {'name': 'TF-IDF + Logistic Regression', 'type': 'traditional'},
    'tfidf_random_forest': {'name': 'TF-IDF + Random Forest', 'type': 'traditional'},
    'transformer_miniLM': {'name': 'Transformer MiniLM', 'type': 'transformer'}
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    model_info = []
    for name, model in models.items():
        model_info.append({
            'id': name,
            'name': model['name'],
            'type': model['type'],
            'description': f"{model['type'].title()} fake news detection model"
        })
    
    return jsonify({'models': model_info})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '').strip()
        model_name = data.get('model', 'tfidf_logistic')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Mock prediction
        fake_keywords = ['shocking', 'breaking', 'unbelievable', 'secret']
        has_fake_keywords = any(keyword in text.lower() for keyword in fake_keywords)
        
        prediction = 'Fake' if has_fake_keywords or len(text) < 50 else 'Real'
        confidence = 0.85 if prediction == 'Fake' else 0.78
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'model_used': models[model_name]['name']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/enhanced', methods=['POST'])
def predict_enhanced():
    try:
        data = request.json
        text = data.get('text', '').strip()
        source = data.get('source', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Mock enhanced prediction
        fake_keywords = ['shocking', 'breaking', 'unbelievable']
        credible_sources = ['reuters.com', 'cnn.com', 'bbc.com']
        
        has_fake_keywords = any(keyword in text.lower() for keyword in fake_keywords)
        source_credible = any(domain in source.lower() for domain in credible_sources) if source else False
        
        prediction = 'Fake' if has_fake_keywords or not source_credible else 'Real'
        confidence = 0.82 if prediction == 'Fake' else 0.76
        
        features = {
            'sentiment_score': 0.5,
            'credibility_score': 0.8 if source_credible else 0.4,
            'emotional_manipulation': 0.7 if has_fake_keywords else 0.2
        }
        
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'features': features
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze_features():
    try:
        data = request.json
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        features = {
            'sentiment_score': 0.5,
            'credibility_score': 0.6,
            'article_length': len(text.split()),
            'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
        
        return jsonify({'features': features})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Fake News Detection API...")
    print("🌐 Open http://localhost:5000 to access the web interface")
    app.run(debug=True, host='0.0.0.0', port=5000)