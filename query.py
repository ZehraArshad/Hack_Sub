from langchain.chat_models import init_chat_model
from langchain_cohere import CohereEmbeddings  

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Init Groq LLM
groq_key = os.getenv("GROQ_API_KEY")
# 3. Set it in the environment manually
# os.environ["GROQ_API_KEY"] = groq_key
llm = init_chat_model("llama3-8b-8192", model_provider="groq", api_key=groq_key)

# 2. Init Cohere Embeddings + Chroma
embedding_function = CohereEmbeddings(
    model="embed-english-v3.0",
   
    )
vector_store = Chroma(
    persist_directory="chroma",
    embedding_function=embedding_function
)

# 3. Define prompt (template from LangChain Hub or custom)
PROMPT_TEMPLATE = """
You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# 4. RAG function
def answer_question(query):
    docs = vector_store.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt_messages = prompt.invoke({"context": context, "question": query})
    response = llm.invoke(prompt_messages)
    print("Answer:", response.content)

# 5. Ask something
if __name__ == "__main__":
    question = input("Ask a question: ")
    answer_question(question)

