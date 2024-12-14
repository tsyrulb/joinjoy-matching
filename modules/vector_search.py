# vector_search.py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import os

class VectorSearch:
    def __init__(self, collection_name, dim=384):
        self.dim = dim
        self.collection_name = collection_name

        # Environment variables for Zilliz Cloud
        zilliz_host = os.environ.get("ZILLIZ_HOST", "in03-92f34bb094d8ac4.serverless.gcp-us-west1.cloud.zilliz.com")
        zilliz_port = os.environ.get("ZILLIZ_PORT", "443")
        zilliz_user = os.environ.get("ZILLIZ_USER", "db_92f34bb094d8ac4")
        zilliz_password = os.environ.get("ZILLIZ_PASSWORD") 

        if not zilliz_password:
            raise ValueError("ZILLIZ_PASSWORD environment variable not set")

        # Connect securely using TLS (secure=True)
        connections.connect(
            alias="default",
            host=zilliz_host,
            port=zilliz_port,
            user=zilliz_user,
            password=zilliz_password,
            secure=True
        )

        fields = [
            FieldSchema(name="UserId", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields, description=f"Collection for {self.collection_name} embeddings")
        self.user_collection = Collection(self.collection_name, schema=schema)

        # Create index if not exists
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        self.user_collection.create_index(field_name="embedding", index_params=index_params)
        self.user_collection.load()

    def insert_user_embeddings(self, user_ids, embeddings):
        if not embeddings:
            print("No embeddings to insert.")
            return

        insert_data = [user_ids, embeddings]
        self.user_collection.insert(insert_data)
        self.user_collection.flush()

    def search_similar_users(self, query_embedding, top_k=50, nprobe=10):
        search_params = {"metric_type": "IP", "params": {"nprobe": nprobe}}
        results = self.user_collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k
        )
        return results[0] if results else []
    
    def delete_user_embedding(self, user_id):
        expr = f"UserId == {user_id}"
        self.user_collection.delete(expr)
        self.user_collection.flush()

    def upsert_user_embedding(self, user_id, embedding):
        self.delete_user_embedding(user_id)
        self.insert_user_embeddings([user_id], [embedding])
        self.user_collection.flush()
