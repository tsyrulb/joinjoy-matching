[![Frontend](https://img.shields.io/badge/GitHub-Frontend-blue?style=for-the-badge)](https://github.com/tsyrulb/join-joy-front)
[![Backend](https://img.shields.io/badge/GitHub-Backend-yellow?style=for-the-badge)](https://github.com/tsyrulb/joinjoy)

---

# Flask API for JoinJoy

This repository contains the Flask-based backend (the "Flask API") of the JoinJoy project, a platform that intelligently recommends activities and users for social outings. It integrates semantic embeddings, vector search, and personalized recommendations.

## Key Features

- **User & Activity Embeddings**: Leverages [sentence-transformers](https://www.sbert.net/) to generate semantic embeddings for users and activities.
- **Vector Search Integration**: Uses [Milvus/Zilliz Cloud](https://zilliz.com/) as a vector database for efficient similarity searches.
- **Semantic Recommendations**: Combines user interest embeddings, TF-IDF analysis, geographic distances, and feedback adjustments to recommend activities and users that genuinely match preferences.
- **Caching & Performance**: Integrates Redis for caching frequently accessed data and results.
- **Data Sources**: Retrieves user, activity, and feedback data from Azure SQL Database. Handles file uploads (like profile photos) via Azure Blob Storage.
![Description of GIF](JoinJoy4.gif)

## Technologies & Libraries

- **Backend**: Python 3.10, [Flask](https://flask.palletsprojects.com/)
- **NLP & Embeddings**: [sentence-transformers](https://www.sbert.net/)
- **Vector Database**: [pymilvus](https://pymilvus.readthedocs.io/en/stable/) for Milvus/Zilliz Cloud
- **Database Access**: [SQLAlchemy](https://www.sqlalchemy.org/) and [pyodbc](https://github.com/mkleehammer/pyodbc) for Azure SQL
- **Caching**: [redis-py](https://github.com/redis/redis-py)
- **Data Processing**: [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/)
- **Geocoding & Distance**: [geopy](https://geopy.readthedocs.io/)

## Prerequisites

- Python 3.10 or above
- A running instance of Azure SQL Database
- Access to Zilliz Cloud (or Milvus) and Redis
- (Optional) Docker for containerized deployment

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/joinjoy-flaskapi.git
   cd joinjoy-flaskapi
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**:

   You need to set environment variables for database connections, APIs, and keys. For example:
   ```bash
   export DB_USER="your_db_user"
   export DB_PASSWORD="your_db_password"
   export DB_SERVER="your_server_name.database.windows.net"
   export DB_NAME="join-joy-db"
   export REDIS_HOST="redis.railway.internal"
   export REDIS_PORT="6379"
   export REDIS_PASSWORD="your_redis_password"
   export NET_CORE_API_BASE_URL="https://webapi.example.com/api/matching"
   # Add any other required environment variables
   ```

   Adjust these according to your configuration.

4. **Database & Vector Setup**:
   - Ensure you have the Azure SQL DB ready and accessible.
   - Ensure Milvus/Zilliz Cloud credentials and endpoints are set within the code or environment.
   - Ensure Redis is reachable with given credentials.

5. **Run Precomputation**:
   The code precomputes embeddings and vector indexes at startup. Just run the app, it will fetch data, compute embeddings, and insert them into Milvus.

6. **Run the Flask API**:
   ```bash
   flask run --host=0.0.0.0 --port=5001
   ```
   
   By default, it runs in development mode. For production, consider using a WSGI server like gunicorn.

## Endpoints

- `GET /recommend_activities?user_id=<id>&top_n=<number>`: Returns top activity recommendations.
- `GET /recommend_users?activity_id=<id>&user_id=<requester_id>&top_n=<number>`: Returns user recommendations for a given activity and requester.
- `POST /find_matches`: Uses semantic key-value matches for interests.
- `POST /refresh_data`: Refreshes cached data and embeddings (useful after updates).

## Docker & Deployment

- **Docker Build**:
  ```bash
  docker build -t yourusername/joinjoy-flaskapi:latest .
  ```
- **Run**:
  ```bash
  docker run -p 5001:5001 --env-file .env yourusername/joinjoy-flaskapi:latest
  ```


## Challenges & Solutions

- **Embedding & Vector Search**: Integrating sentence-transformers with a vector database was challenging. Solved by careful encoding and indexing.
- **Database Queries**: Ensuring efficient queries to Azure SQL and caching results in Redis improved performance.
- **Semantic Recommendations**: Balancing semantic similarity, location, and user feedback required iterative refinement and testing.

```
