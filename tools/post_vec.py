from pgvector.psycopg import register_vector
import psycopg
import numpy as np

conn = psycopg.connect(dbname='testdb', autocommit=True)



conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
register_vector(conn)

conn.execute('DROP TABLE IF EXISTS my_vector_table')
conn.execute('CREATE TABLE my_vector_table (id bigserial PRIMARY KEY, cam_id integer, track_id integer, embedding vector(3))')
vectors = []  # List of vectors to insert

# Example vectors
vectors.append(np.array([1, 2, 3]))  # Vector 1
vectors.append(np.array([4, 5, 6]))

for i in range(2):
    for j in range(2):
        vector = vectors[i]
        query = 'INSERT INTO my_vector_table (cam_id, track_id, embedding) VALUES (%s, %s, %s)'
        conn.execute(query, (i, j, vector))

conn.execute('CREATE INDEX ON my_vector_table USING ivfflat (embedding vector_cosine_ops)')

query_vector = np.array([4, 5, 7])

cam_id = 1
query = ' SELECT id, track_id, embedding, embedding::vector <-> %s::vector AS distance FROM my_vector_table WHERE cam_id != %s ORDER BY embedding <-> %s LIMIT 1'
result = conn.execute(query, (query_vector, cam_id, query_vector)).fetchall()

print(result)
