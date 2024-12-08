from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import numpy as np

class VectorSearch:
    def __init__(self, collection_name, host="localhost", port="19530", dim=384):
        self.dim = dim
        self.collection_name = collection_name
        connections.connect("default", host=host, port=port)

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
        # embeddings should be a list of lists (each list is one embedding)
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
        return results[0]  # Return list of hits
    
    def delete_user_embedding(self, user_id):
        expr = f"UserId == {user_id}"
        self.user_collection.delete(expr)
        self.user_collection.flush()

    def upsert_user_embedding(self, user_id, embedding):
        # First delete old one
        self.delete_user_embedding(user_id)
        # Then re-insert the new embedding
        self.user_collection.insert([[user_id], [embedding]])
        self.user_collection.flush()
