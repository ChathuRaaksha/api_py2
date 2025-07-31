from flask import Flask, request, jsonify
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)

# Dummy destination data
destinations_data = [
    {"id": 1, "name": "Vasa Museum", "category": "Culture", "location": "Stockholm", "description": "A museum dedicated to the 17th-century ship, the Vasa.", 
     "food_and_drink": 4, "culture_and_heritage": 9, "nature_and_adventure": 5, "art_and_creativity": 7, "wellness_and_relaxation": 4, 
     "sustainable_travel": 6, "urban_exploration": 8, "community_and_social_experiences": 7},
    
    {"id": 2, "name": "Pizza Hut T-Central", "category": "Food & Drink", "location": "Stockholm", "description": "Popular fast food chain, great for pizza.", 
     "food_and_drink": 9, "culture_and_heritage": 4, "nature_and_adventure": 3, "art_and_creativity": 2, "wellness_and_relaxation": 5, 
     "sustainable_travel": 7, "urban_exploration": 6, "community_and_social_experiences": 6},
    
    {"id": 3, "name": "Skansen", "category": "Nature & Adventure", "location": "Stockholm", "description": "An open-air museum with historical buildings and animals.", 
     "food_and_drink": 3, "culture_and_heritage": 6, "nature_and_adventure": 9, "art_and_creativity": 4, "wellness_and_relaxation": 6, 
     "sustainable_travel": 8, "urban_exploration": 3, "community_and_social_experiences": 8},
    
    {"id": 4, "name": "ABBA Museum", "category": "Culture", "location": "Stockholm", "description": "Museum dedicated to the famous Swedish pop group ABBA.", 
     "food_and_drink": 6, "culture_and_heritage": 10, "nature_and_adventure": 2, "art_and_creativity": 10, "wellness_and_relaxation": 3, 
     "sustainable_travel": 4, "urban_exploration": 7, "community_and_social_experiences": 9},
    
    {"id": 5, "name": "Fjäderholmarna Islands", "category": "Nature & Adventure", "location": "Stockholm", "description": "A group of small islands great for hiking and seafood.", 
     "food_and_drink": 7, "culture_and_heritage": 7, "nature_and_adventure": 10, "art_and_creativity": 5, "wellness_and_relaxation": 9, 
     "sustainable_travel": 8, "urban_exploration": 5, "community_and_social_experiences": 7},
    
    {"id": 6, "name": "Södra Teatern", "category": "Urban Exploration", "location": "Stockholm", "description": "A popular venue for concerts, arts, and nightlife in the city.", 
     "food_and_drink": 8, "culture_and_heritage": 8, "nature_and_adventure": 6, "art_and_creativity": 9, "wellness_and_relaxation": 7, 
     "sustainable_travel": 5, "urban_exploration": 9, "community_and_social_experiences": 8}
]

# Convert the destinations data into a DataFrame
destinations = pd.DataFrame(destinations_data)

# Function to calculate match percentage using SVM
def calculate_match_percentage(user_preferences, destinations):
    features = ['food_and_drink', 'culture_and_heritage', 'nature_and_adventure', 
                'art_and_creativity', 'wellness_and_relaxation', 'sustainable_travel', 
                'urban_exploration', 'community_and_social_experiences']
    
    X = destinations[features]
    match_scores = []

    for _, row in X.iterrows():
        differences = [abs(user_preferences[feature] - row[feature]) for feature in features]
        match_score = 10 - (sum(differences) / len(features))
        match_scores.append(match_score)

    y = match_scores
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm_model = SVR(kernel='linear')
    svm_model.fit(X_train_scaled, y_train)
    
    preferences = pd.DataFrame(user_preferences, index=[0])
    preferences_scaled = scaler.transform(preferences)
    predicted_match = svm_model.predict(preferences_scaled)
    
    predicted_match_percentage = predicted_match[0] * 10
    
    destinations['match_score'] = svm_model.predict(scaler.transform(destinations[features])) * 10
    recommendations = destinations[['name', 'match_score', 'category', 'location', 'description']]
    recommendations = recommendations.sort_values(by='match_score', ascending=False)
    
    recommendations = recommendations[recommendations['match_score'] > 65]
    
    return recommendations, predicted_match_percentage

# API endpoint to get recommendations based on user preferences
@app.route('/get_recommendations', methods=['GET'])
def get_recommendations():
    try:
        user_preferences = {
            'food_and_drink': int(request.args.get('food_and_drink')),
            'culture_and_heritage': int(request.args.get('culture_and_heritage')),
            'nature_and_adventure': int(request.args.get('nature_and_adventure')),
            'art_and_creativity': int(request.args.get('art_and_creativity')),
            'wellness_and_relaxation': int(request.args.get('wellness_and_relaxation')),
            'sustainable_travel': int(request.args.get('sustainable_travel')),
            'urban_exploration': int(request.args.get('urban_exploration')),
            'community_and_social_experiences': int(request.args.get('community_and_social_experiences'))
        }

        # Get recommendations and predicted match score
        recommendations, predicted_match = calculate_match_percentage(user_preferences, destinations)

        recommendation_list = recommendations.to_dict(orient='records')

        return jsonify({
            'predicted_match_score': f"{predicted_match:.2f}%",
            'recommendations': recommendation_list
        })

    except Exception as e:
        print(f"Error: {str(e)}")  # Print the error
        return jsonify({"error": str(e)}), 500

# Hello World function for testing
@app.route('/hello', methods=['GET'])
def hello_world():
    return jsonify({"message": "Hello, World!"})

if __name__ == "__main__":
    app.run(debug=True)
