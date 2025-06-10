from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import os
from serpapi import GoogleSearch
import openai
from dotenv import load_dotenv
import re
import json
import hashlib
import logging
from pathlib import Path
import io
import requests
import stripe
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Load precomputed data
faiss_index = faiss.read_index("faiss_index.index")
image_embeddings = np.load("image_embeddings.npy")
with open("id_to_caption.json", "r") as f:
    id_to_caption = json.load(f)

# Constants
MAX_PRODUCTS = 12
CACHE_FILE = Path("blip_caption_cache.json")
USER_PROFILES_FILE = Path("user_profiles.json")
PERSONALIZATION_WEIGHTS = {
    'price': 0.3,
    'brand': 0.25,
    'style': 0.2,
    'color': 0.15,
    'category': 0.1
}
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM_INSTRUCTIONS = "You are a helpful shopping assistant who helps users refine product searches based on their input. Your goal is to interpret user messages and update search filters like color, category, price, style, or brand."

# Initialize Stripe
import os
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")  # ✅ GOOD

'''# Load BLIP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)'''

# Data structures
if CACHE_FILE.exists():
    caption_cache = json.loads(CACHE_FILE.read_text())
else:
    caption_cache = {}

if USER_PROFILES_FILE.exists():
    user_profiles = json.loads(USER_PROFILES_FILE.read_text())
else:
    user_profiles = {}

user_carts = defaultdict(list)
user_wishlists = defaultdict(list)
user_view_history = defaultdict(list)

class PersonalizationEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def update_user_profile(self, user_id, product_data, action_type='add_to_cart'):
        """Update user profile based on their interactions"""
        if user_id not in user_profiles:
            user_profiles[user_id] = {
                'preferences': defaultdict(float),
                'purchase_history': [],
                'view_history': [],
                'last_updated': datetime.now().isoformat()
            }
        
        # Weight actions differently
        weight = 1.0
        if action_type == 'view':
            weight = 0.5
        elif action_type == 'wishlist':
            weight = 0.8
            
        # Extract features from product
        features = {
            'price': self._normalize_price(product_data.get('price', '0')),
            'brand': product_data.get('source', '').lower(),
            'style': self._extract_style(product_data.get('title', '')),
            'color': self._extract_color(product_data.get('title', '')),
            'category': self._extract_category(product_data.get('title', ''))
        }
        
        # Update preferences with weighted values
        for feature, value in features.items():
            if value:
                user_profiles[user_id]['preferences'][feature] = (
                    user_profiles[user_id]['preferences'].get(feature, 0) * 0.7 + 
                    PERSONALIZATION_WEIGHTS[feature] * weight
                )
                
        # Record action
        if action_type == 'add_to_cart':
            user_profiles[user_id]['purchase_history'].append({
                'product': product_data,
                'timestamp': datetime.now().isoformat()
            })
        elif action_type == 'view':
            user_profiles[user_id]['view_history'].append({
                'product': product_data,
                'timestamp': datetime.now().isoformat()
            })
            
        # Save updated profile
        USER_PROFILES_FILE.write_text(json.dumps(user_profiles))
        
    def get_personalized_recommendations(self, user_id, products):
        """Reorder products based on user preferences"""
        if user_id not in user_profiles or not products:
            return products
            
        profile = user_profiles[user_id]['preferences']
        if not profile:
            return products
            
        # Score each product based on user preferences
        scored_products = []
        for product in products:
            score = 0
            product_features = {
                'price': self._normalize_price(product.get('price', '0')),
                'brand': product.get('source', '').lower(),
                'style': self._extract_style(product.get('title', '')),
                'color': self._extract_color(product.get('title', '')),
                'category': self._extract_category(product.get('title', ''))
            }
            
            for feature, value in product_features.items():
                if value and feature in profile:
                    # Simple cosine similarity for text features
                    if feature in ['brand', 'style', 'color', 'category']:
                        vectors = self.vectorizer.fit_transform([str(profile.get(feature, '')), str(value)])
                        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                        score += similarity * profile[feature]
                    # For price, use inverse distance
                    elif feature == 'price':
                        price_diff = abs(profile.get('price', 0) - value)
                        if price_diff == 0:
                            score += PERSONALIZATION_WEIGHTS['price']
                        else:
                            score += PERSONALIZATION_WEIGHTS['price'] / (1 + price_diff)
            
            scored_products.append((score, product))
        
        # Sort by score descending
        scored_products.sort(key=lambda x: x[0], reverse=True)
        return [product for score, product in scored_products]
    
    def _normalize_price(self, price_str):
        """Convert price string to numerical value"""
        try:
            return float(re.sub(r'[^\d.]', '', price_str))
        except:
            return 0
            
    def _extract_style(self, title):
        """Extract style from product title"""
        styles = ['casual', 'formal', 'sport', 'business', 'party', 'beach', 'evening']
        for style in styles:
            if style in title.lower():
                return style
        return ''
        
    def _extract_color(self, title):
        """Extract color from product title"""
        colors = ['red', 'blue', 'green', 'black', 'white', 'yellow', 'pink', 'purple']
        for color in colors:
            if color in title.lower():
                return color
        return ''
        
    def _extract_category(self, title):
        """Extract category from product title"""
        categories = ['shirt', 'pant', 'dress', 'shoe', 'jacket', 'hat', 'accessory']
        for category in categories:
            if category in title.lower():
                return category
        return ''

personalization_engine = PersonalizationEngine()

def hash_image(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

def get_fashion_caption(image_bytes):
    img_hash = hash_image(image_bytes)
    if img_hash in caption_cache:
        logging.info("Using cached caption.")
        return caption_cache[img_hash]

    try:
        # Load and process image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            query_embedding = clip_model.get_image_features(**inputs).cpu().numpy().astype("float32")

        # Search for nearest neighbor
        D, I = faiss_index.search(query_embedding, k=1)
        top_idx = I[0][0]
        matched_caption = id_to_caption[str(top_idx)]["caption"]

        caption_cache[img_hash] = matched_caption
        CACHE_FILE.write_text(json.dumps(caption_cache))
        logging.info(f"Matched caption: {matched_caption}")
        return matched_caption

    except Exception as e:
        logging.error(f"Matching error: {e}")
        return "Unknown fashion item"

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        image_bytes = request.files['image'].read()
        caption = get_fashion_caption(image_bytes)

        # Get optional filters from form data (for Preferences)
        filters = request.form.get('filters')
        if filters:
            filters = json.loads(filters)
        else:
            filters = {}

        # Get user ID
        user_id = request.form.get('user_id', 'default')

        # Build query using filters + caption
        query = " ".join(str(value) for value in filters.values()) + " " + caption

        # Build product search payload
        product_payload = {
            "query": query,
            "filters": filters,
            "user_id": user_id
        }

        # Call product search inline
        with app.test_request_context():
            product_request = app.test_client().post("/get-products", json=product_payload)
            product_data = product_request.get_json()

        return jsonify({
            'description': caption,
            'message': f"I found this description: '{caption}'. Based on your preferences, here are some results.",
            'products': product_data.get("products", []),
            'appliedFilters': filters
        })

    except Exception as e:
        logging.error("Image processing error:", exc_info=True)
        return jsonify({'error': 'Failed to process image. Please try another image.'}), 500

@app.route("/get-products", methods=["POST"])
def get_products():
    data = request.get_json()
    query = data.get('query', '').strip()
    filters = data.get('filters', {})
    user_id = data.get('user_id', 'default')

    if not query:
        return jsonify({"error": "Please provide a search query"}), 400

    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": os.getenv("SERPAPI_KEY"),
        "num": MAX_PRODUCTS,
        "gl": "in",
        "hl": "en"
    }

    if 'maxPrice' in filters:
        try:
            params["price_to"] = int(float(filters['maxPrice']))
        except (ValueError, TypeError):
            logging.warning("Invalid price filter")

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        shopping_results = results.get("shopping_results", [])[:MAX_PRODUCTS]

        products = []
        for item in shopping_results:
            product_link = item.get("offers", {}).get("link", item.get("link", "#"))
            products.append({
                "title": item.get("title", "Product"),
                "link": product_link,
                "image": item.get("thumbnail", ""),
                "price": item.get("price", "Price not available"),
                "source": item.get("source", ""),
                "merchant": item.get("merchant", "")
            })

        # Personalize results if user is logged in
        personalized_products = personalization_engine.get_personalized_recommendations(user_id, products)

        return jsonify({
            "products": personalized_products,
            "personalized": user_id != 'default'
        })

    except Exception as e:
        logging.error("Product search error:", exc_info=True)
        return jsonify({"error": "Failed to fetch products"}), 500

# --- Helper functions ---

def generate_prompt(user_message, current_caption):
    return f"""
You are a smart shopping assistant.

The user uploaded a product image with the caption: "{current_caption}"

Now, the user typed: "{user_message}"

Your job is to extract updated **search filters** that describe what the user wants to see. Only return filters that are present or clearly implied. These include:

- brand (e.g. Puma, Adidas)
- color (e.g. black, red)
- style (e.g. round neck, crop top)
- category (e.g. t-shirt, shoes)
- gender (e.g. men, women, unisex)
- maxPrice (if mentioned like "under 800", "below 1000")

Output only a valid JSON object with relevant keys, nothing else.

Examples:

Caption: "Nike Red Running Shoes"
User: "Show me Adidas black ones"
Output:
{{ "brand": "Adidas", "color": "black", "category": "shoes" }}

Caption: "Zara women's blue summer dress"
User: "Any red long sleeve tops under 800"
Output:
{{ "color": "red", "style": "long sleeve", "category": "top", "maxPrice": "800" }}

Now do the same for:

Caption: "{current_caption}"
User: "{user_message}"

JSON output:
"""

def parse_filters_from_response(response_text):
    try:
        return json.loads(response_text)
    except Exception as e:
        logging.warning("Failed to parse filters JSON: %s", e)
        return {}

def build_search_query(filters):
    # Prioritize filters over old caption
    parts = []
    for key in ['brand', 'color', 'style', 'category']:
        if key in filters and filters[key]:
            parts.append(filters[key])
    return " ".join(parts).strip() or "fashion clothing"

# --- Flask route ---

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        current_filters = data.get('filters', {})
        current_caption = data.get('current_caption', '')
        user_id = data.get('user_id', 'default')

        # Create GPT prompt
        prompt = generate_prompt(user_message, current_caption)

        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openai/gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful shopping assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_text = response.json()["choices"][0]["message"]["content"]

        updated_filters = parse_filters_from_response(response_text)
        combined_filters = {**current_filters, **updated_filters}

        # Build refined search query
        search_query = build_search_query(combined_filters)

        # Prepare SerpAPI request
        params = {
            "engine": "google_shopping",
            "q": search_query,
            "api_key": os.getenv("SERPAPI_KEY"),
            "num": MAX_PRODUCTS,
            "gl": "in",
            "hl": "en"
        }

        if 'maxPrice' in combined_filters:
            try:
                params['price_to'] = int(float(combined_filters['maxPrice']))
            except ValueError:
                logging.warning("Invalid price filter format")

        search = GoogleSearch(params)
        results = search.get_dict()
        shopping_results = results.get("shopping_results", [])[:MAX_PRODUCTS]

        # Format product output
        products = []
        for item in shopping_results:
            link = item.get("offers", {}).get("link", item.get("link", "#"))
            products.append({
                "title": item.get("title", "Product"),
                "link": link,
                "image": item.get("thumbnail", ""),
                "price": item.get("price", "N/A"),
                "source": item.get("source", ""),
                "merchant": item.get("merchant", "")
            })

        # Personalize (if you have logic for it)
        personalized_products = personalization_engine.get_personalized_recommendations(user_id, products)

        return jsonify({
            "response": response_text,
            "filters": combined_filters,
            "products": personalized_products,
            "query_used": search_query
        })

    except Exception as e:
        logging.error("Error in /chat route:", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/add-to-cart', methods=['POST'])
def add_to_cart():
    try:
        data = request.json
        user_id = data.get('user_id', 'default')
        product = data.get('product')
        
        if not product:
            return jsonify({"error": "No product provided"}), 400

        # Add product to cart
        user_carts[user_id].append(product)
        
        # Update user profile with this interaction
        personalization_engine.update_user_profile(user_id, product, 'add_to_cart')
        
        # Generate personalized recommendations
        similar_products = get_similar_products(product)
        personalized_recs = personalization_engine.get_personalized_recommendations(user_id, similar_products)
        
        return jsonify({
            "success": True,
            "message": f"Added {product['title']} to cart",
            "cart_count": len(user_carts[user_id]),
            "personalized_recommendations": personalized_recs[:3],  # Return top 3 recommendations
            "complementary_items": get_complementary_items(product)[:2]  # Items that go well with this product
        })

    except Exception as e:
        logging.error("Add to cart error:", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/get-cart', methods=['GET'])
def get_cart():
    try:
        user_id = request.args.get('user_id', 'default')
        cart = user_carts.get(user_id, [])
        
        return jsonify({
            "success": True,
            "cart": cart,
            "cart_count": len(cart)
        })

    except Exception as e:
        logging.error("Get cart error:", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/create-checkout-session", methods=["POST"])
def create_checkout_session():
    try:
        data = request.json
        product = data.get('product')
        user_id = data.get('user_id', 'default')
        
        if not product:
            return jsonify({"error": "No product provided"}), 400

        # Convert price to cents (Stripe requires integer amount in smallest currency unit)
        try:
            # Remove any currency symbols and convert to float
            price = float(re.sub(r'[^\d.]', '', product['price']))
            amount = int(price * 100)  # Convert to cents/paisa
        except (ValueError, TypeError):
            return jsonify({"error": "Invalid price format"}), 400

        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'inr',
                    'product_data': {
                        'name': product['title'],
                        'images': [product['image']] if product.get('image') else [],
                    },
                    'unit_amount': amount,
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url='http://localhost:3000/success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url='http://localhost:3000/cancel',
            metadata={
                'product_id': product.get('product_id', ''),
                'title': product['title'],
                'user_id': user_id
            }
        )

        return jsonify({
            "success": True,
            "url": session.url
        })

    except Exception as e:
        logging.error("Checkout error:", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/create-cart-checkout-session", methods=["POST"])
def create_cart_checkout_session():
    try:
        data = request.json
        user_id = data.get('user_id', 'default')
        cart = user_carts.get(user_id, [])
        
        if not cart:
            return jsonify({"error": "Cart is empty"}), 400

        line_items = []
        for product in cart:
            try:
                price = float(re.sub(r'[^\d.]', '', product['price']))
                amount = int(price * 100)
                line_items.append({
                    'price_data': {
                        'currency': 'inr',
                        'product_data': {
                            'name': product['title'],
                            'images': [product['image']] if product.get('image') else [],
                        },
                        'unit_amount': amount,
                    },
                    'quantity': 1,
                })
            except (ValueError, TypeError):
                continue  # Skip invalid products

        if not line_items:
            return jsonify({"error": "No valid products in cart"}), 400

        session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=line_items,
            mode='payment',
            success_url='http://localhost:3000/success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url='http://localhost:3000/cancel',
            metadata={
                'checkout_type': 'cart',
                'user_id': user_id
            }
        )

        # Clear cart after successful checkout creation
        user_carts[user_id] = []

        return jsonify({
            "success": True,
            "url": session.url
        })

    except Exception as e:
        logging.error("Cart checkout error:", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/get-user-profile', methods=['GET'])
def get_user_profile():
    user_id = request.args.get('user_id', 'default')
    if user_id in user_profiles:
        return jsonify({
            "success": True,
            "profile": user_profiles[user_id]
        })
    return jsonify({
        "success": False,
        "message": "No profile found"
    })

@app.route('/get-personalized-products', methods=['GET'])
def get_personalized_products():
    user_id = request.args.get('user_id', 'default')
    category = request.args.get('category', '')
    
    # Get base products
    params = {
        "engine": "google_shopping",
        "q": category or "fashion clothing",
        "api_key": os.getenv("SERPAPI_KEY"),
        "num": MAX_PRODUCTS,
        "gl": "in",
        "hl": "en"
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        products = [{
            "title": item.get("title", "Product"),
            "link": item.get("offers", {}).get("link", item.get("link", "#")),
            "image": item.get("thumbnail", ""),
            "price": item.get("price", "Price not available"),
            "source": item.get("source", ""),
            "merchant": item.get("merchant", "")
        } for item in results.get("shopping_results", [])[:MAX_PRODUCTS]]
        
        # Personalize the results
        personalized_products = personalization_engine.get_personalized_recommendations(user_id, products)
        
        return jsonify({
            "success": True,
            "products": personalized_products,
            "personalization_score": user_profiles.get(user_id, {}).get('preferences', {})
        })
    except Exception as e:
        logging.error("Personalized products error:", exc_info=True)
        return jsonify({"error": str(e)}), 500

def get_similar_products(product):
    """Find similar products based on current product"""
    params = {
        "engine": "google_shopping",
        "q": product.get('title', ''),
        "api_key": os.getenv("SERPAPI_KEY"),
        "num": MAX_PRODUCTS,
        "gl": "in",
        "hl": "en"
    }
    
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        return [{
            "title": item.get("title", "Product"),
            "link": item.get("offers", {}).get("link", item.get("link", "#")),
            "image": item.get("thumbnail", ""),
            "price": item.get("price", "Price not available"),
            "source": item.get("source", ""),
            "merchant": item.get("merchant", "")
        } for item in results.get("shopping_results", [])[:MAX_PRODUCTS]]
    except:
        return []

def get_complementary_items(product):
    """Find complementary items using AI-generated suggestions"""
    title = product.get('title', '')
    caption = product.get('caption', '')  # Make sure to pass the CLIP-generated caption
    if not title and not caption:
        return []

    try:
        # Generate complementary product suggestions using OpenRouter
        prompt = f"""Given this fashion item: "{title if title else caption}", suggest 3-5 specific complementary fashion items that would go well with it. 
        Return ONLY a comma-separated list of product names, nothing else.
        Examples:
        - If input is "white formal shirt", output: "black dress pants, leather belt, silk tie, formal shoes"
        - If input is "summer floral dress", output: "strappy sandals, woven handbag, sun hat, denim jacket"
        - If input is "running shoes", output: "athletic socks, sports shorts, moisture-wicking t-shirt, running armband\"""" 

        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openai/gpt-3.5-turbo",  # or use a better model
            "messages": [
                {"role": "system", "content": "You are a fashion expert who suggests complementary clothing items."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        response = requests.post(OPENROUTER_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_text = response.json()["choices"][0]["message"]["content"]

        # Parse the response into individual queries
        complementary_queries = [q.strip() for q in response_text.split(",") if q.strip()]
        
        # Get products for each complementary query
        complementary_products = []
        for query in complementary_queries[:3]:  # Limit to top 3 suggestions
            params = {
                "engine": "google_shopping",
                "q": f"{query} fashion",
                "api_key": os.getenv("SERPAPI_KEY"),
                "num": 1,  # Get just 1 product per complementary item
                "gl": "in",
                "hl": "en"
            }
            
            try:
                search = GoogleSearch(params)
                results = search.get_dict()
                if results.get("shopping_results"):
                    item = results["shopping_results"][0]
                    complementary_products.append({
                        "title": item.get("title", query),
                        "link": item.get("offers", {}).get("link", item.get("link", "#")),
                        "image": item.get("thumbnail", ""),
                        "price": item.get("price", "Price not available"),
                        "source": item.get("source", ""),
                        "merchant": item.get("merchant", ""),
                        "complementary_to": title if title else caption
                    })
            except Exception as e:
                logging.error(f"Failed to search for complementary item {query}: {str(e)}")
                continue

        return complementary_products[:5]  # Return max 3 complementary items

    except Exception as e:
        logging.error(f"Error in get_complementary_items: {str(e)}")
        return []

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=bool(os.getenv("FLASK_DEBUG", False)))