import cohere
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

# Initialize Cohere client (v2)
co = cohere.ClientV2(api_key)

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def main():
    word1 = "samsung"
    word2 = "macOS"

    # Get embeddings
    response1 = co.embed(
        texts=[word1, word2],
        model="embed-english-v3.0",  # or embed-v4.0 if needed
        input_type="classification",
        embedding_types=["float"],
    )
    # response2 = co.embed(
    #     texts=[word2],
    #     model="embed-english-v3.0",
    #     input_type="classification",
    #     embedding_types=["float"],
    # )

    # Extract float embeddings
    embedding1 = response1.embeddings.float[0]
    embedding2 = response1.embeddings.float[1]

    # Print vector details
    print(f"Vector length: {len(embedding1)}")

    # Cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)
    print(f"Cosine similarity between '{word1}' and '{word2}': {similarity:.4f}")

if __name__ == "__main__":
    main()

