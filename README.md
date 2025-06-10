# 🛍️ StyleFinder: AI-Powered Personal Shopping Assistant

**Finale Submission - IIT-M AI Hackathon 2025**  
**Theme**: *AI-Powered E-Commerce*

---

## 🚀 Overview

**StyleFinder** is not just another shopping tool — it’s a *revolution* in how users discover fashion. Powered by cutting-edge AI, StyleFinder transforms your image inspiration into personalized shopping experiences. Whether you have a fashion photo or a vibe in mind, StyleFinder brings the perfect products to your fingertips with real-time search, visual AI, and natural interaction.

> 🧠 Multimodal, preference-aware, and chat-driven — *the ultimate AI shopping companion.*

---

## 🔧 Key Features

- 📸 **Image-Based Product Discovery**  
  Upload any fashion item image to instantly discover similar and visually matching products.

- 🧠 **Visual Intelligence via CLIP**  
  Leverages OpenAI’s CLIP model to convert images into rich semantic embeddings, which are matched against catalog descriptions.

- 🧲 **FAISS-Powered Fast Similarity Matching**
  Utilizes FAISS for high-performance vector similarity search, enabling rapid identification of the most relevant catalog entries.

- 🔍 **Real-Time Product Search with SerpAPI**  
  Converts the most relevant catalog description into a search query and uses SerpAPI to fetch real-world, shoppable product links.

- 💬 **Smart Chatbot Interface**  
  GPT-powered chatbot refines results, answers queries, and enhances interactivity.

- 🎨 **Preference-Aware Personalization**
  Users can input fashion preferences such as style, brand, or color to guide recommendations at every interaction stage.

- 🛒 **Intelligent Cart Integration**  
  Add your favorite items to the cart and enjoy:
  	- Automatic suggestions for complementary products to complete your look
	- Smarter and more relevant recommendations, refined using the cart context

- 💳 **Secure Checkout with Stripe**  
  Real or simulated payments through Stripe APIs.

- 🎨 **Personalization Tab**  
  Feed your preferences and get tailor-made recommendations.

---

## 🧱 System Architecture
```mermaid
graph TD
  A[User Uploads Image] --> B[Pretrained CLIP Image Embedding]
  A2[User Preferences or Profile] --> B

  subgraph Vector_Matching_Engine
    D1[Preprocessed Catalog - Text and Vectors]
    D3[CLIP Text Embeddings of Catalog Items]
    D2[FAISS Index for Similarity Search]
    B --> D2
    D1 --> D3
    D3 --> D2
    D2 --> C[Top-k Similar Items from Catalog]
  end

  C --> E[Most Relevant Product Description]
  E --> F[Fetch Matches via SerpAPI]
  F --> G[Display Recommendations on Frontend]

  G --> H[Chatbot using GPT for Feedback]
  H --> G
  H --> I[Modify Search Query or Filters]

  G --> J[🛒 Add to Cart]
  J --> K[Stripe Checkout]

  %% Personalization and Cart Intelligence
  J --> L[Trigger Personalization Engine]
  L --> M[Complementary Product Suggestions]
  L --> N[Refined Recommendations<br>based on Cart Context]
  M --> G
  N --> G

%% Clickable node references (for supported environments like Mermaid Live Editor or Docs)
click B href "https://openai.com/research/clip" _blank
click D2 href "https://github.com/facebookresearch/faiss" _blank
click F href "https://serpapi.com/" _blank
click H href "https://platform.openai.com/" _blank
```

---

## 🤖 AI & Tools Used

| Component              | Tool/Model Used             |
|-----------------------|-----------------------------|
| Vision Embedding      | OpenAI CLIP                 |
| Preferences Integration | Vector-weighted relevance   |
| Description Search    | FAISS            |
| Web Search API        | SerpAPI                     |
| Chatbot               | GPT API                     |
| Backend               | Flask (Python)              |
| Frontend              | React.js                    |
| Data Storage          | JSON, Numpy                 |
| Checkout              | Stripe                      |

---

## 🧠 AI Model & Intelligence Layer

The intelligence behind **StyleFinder** is powered by a combination of state-of-the-art AI models and smart engineering for real-time fashion and product recommendations:

### 1. 🖼️ **CLIP (Contrastive Language-Image Pretraining) by OpenAI**

We use CLIP to extract visual semantics from user-uploaded images. It transforms images into high-dimensional embeddings that capture both visual features and their conceptual meaning. These embeddings are crucial for matching the image with similar items from our product catalog.

* **Why CLIP?**
  Unlike traditional CNNs, CLIP understands the *semantic* content of an image—e.g., recognizing that a red sleeveless dress and a maroon bodycon dress are similar in function and style.

### 2. 🧭 **FAISS (Facebook AI Similarity Search)**

To efficiently find the closest product match, we use **FAISS**, an optimized similarity search library. After converting both the image and catalog entries into embeddings, FAISS performs a **cosine similarity** search to find the most relevant catalog item in milliseconds.

* **Use Case**: Enables scalable, lightning-fast retrieval from large product databases.

### 3. 🌐 **SerpAPI for Real-World Product Discovery**

Once we identify the most semantically similar catalog description, we query **SerpAPI** to fetch live product results from the web. This bridges the gap between local inference and dynamic global inventory.

* **Use Case**: Converts a static product catalog into a dynamic discovery engine.

### 4. 💬 **GPT-based Chatbot**

Our chatbot leverages the **GPT API** to offer contextual and conversational support. Users can refine their searches (e.g., “Show cheaper alternatives” or “Suggest something more casual”) and receive smart, natural-language responses.

* **Role of GPT**: Acts as an interactive fashion advisor that personalizes the shopping experience through dialogue.

---

This section gives reviewers and readers a solid technical understanding of how your AI stack contributes to the product. If you want, I can also add a visual summary (like a model interaction diagram) or break this up with icons and callouts for clarity.

---
## ⚙️ Setup Instructions

### 🔧 Quick Start

Clone and configure the repository:
```bash
git clone https://github.com/bharatnageshr/fashion_assistant.git
cd fashion_assistant

# Set environment variables
export SERP_API_KEY=your_serpapi_key
export STRIPE_SECRET_KEY=your_stripe_key
export OPENAI_API_KEY=your_openaiapi_key
export FLASK_DEBUG=True
export OPENROUTER_API_KEY=your_openrouterapi_key

# On Windows:
set SERP_API_KEY=your_serpapi_key
set STRIPE_SECRET_KEY=your_stripe_key
set OPENAI_API_KEY=your_openaiapi_key
set FLASK_DEBUG=True
set OPENROUTER_API_KEY=your_openrouterapi_key
```

---

## 📦 Install Dependencies

**Prerequisites:** Python 3.8+, Node.js, pip, npm

### 🔙 Backend:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 💻 Frontend:
> 💡 Make sure you have Node.js (v16+) and npm installed.
```bash
cd ../frontend
npm install
```

---

## Dataset used for Vectorisation & Fine Tuning
'''
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small?resource=download
'''

---

## ▶️ Run the Application

### Backend:
```bash
cd backend
python app.py
```

### Frontend:
```bash
cd ../frontend
npm start
```
### Open StyleFinderAI
'''
Go to:
http://localhost:5001
'''

---

## 💡 How It Works

1. **Upload** an image on the homepage.
2. Backend generates a **CLIP embedding** and considers **user preferences**.
3. Uses **cosine similarity** to find closest matches in our catalog.
4. Extracted **description** is passed to **SerpAPI** for real-world results.
5. The frontend displays the results and lets users interact with a **chatbot**.
6. Products can be **added to cart**.
7. Payments are handled via **Stripe Checkout**.

---

## 🧪 Example Use Cases

- 👗 Upload a photo of a dress → Get similar dresses + accessories.
- 👟 Upload sneaker image → Discover alternatives + gym accessories.
- 💬 Ask chatbot: “Show cheaper alternatives” or “Match this with jeans.”

---

## 🗂️ Project Structure
```
backend/
 ├── app.py
 └── recommender/
     ├── data/
     │   ├── catalog.json
     │   ├── embeddings.npy
     │   ├── faiss_index.index
     │   └── build_catalog.py
     └── model_utils.py

frontend/
 └── src/
     ├── components/
     │   ├── App.js
     │   ├── Preferences.js
     │   └── Chatbot.js
     └── public/

README.md
```

---

## ✨ Highlights

- ✅ *Multimodal intelligence* (image + preferences)
- ✅ *Search + recommendation* fusion
- ✅ *AI-powered chat refinement*
- ✅ *End-to-end product interaction*: Upload → Recommend → Cart → Checkout
- ✅ *Fast, responsive frontend and scalable backend*

---

## 👥 Team

- **Reddi Srujan**  
- **Bharat Nagesh**

> Built with 💖 at IIT-M AI Hackathon 2025

