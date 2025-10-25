import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def create_documents_from_df(df, source_name):
    """Convert DataFrame rows to LangChain Document objects"""
    documents = []
    
    # Add a summary document for each dataset
    if 'crop' in df.columns:  # Gujarat crops
        summary = f"Crops grown in Gujarat: {', '.join(df['crop'].tolist())}\n"
        summary += f"Total crops: {len(df)}\n"
        documents.append(Document(
            page_content=summary,
            metadata={"source": source_name, "type": "summary"}
        ))
    
    # Then add individual rows
    for idx, row in df.iterrows():
        text_content = "\n".join([
            f"{col}: {value}" 
            for col, value in row.items()
        ])
        
        doc = Document(
            page_content=text_content,
            metadata={
                "source": source_name,
                "row_index": idx
            }
        )
        documents.append(doc)
        
        if (idx + 1) % 500 == 0:
            print(f"   Processed {idx + 1} rows...", end="\r")
    
    return documents

print("🔨 CREATING VECTOR STORE\n")

CLEANED_FILES = [
    ("data/rainfall_subdivisional_clean.csv", "Rainfall Subdivision"),
    ("data/wheat_punjab_clean.csv", "Wheat Punjab"),
    ("data/rice_karnataka_clean.csv", "Rice Karnataka"),
    ("data/crops_gujarat_clean.csv", "Gujarat Crops")
]

docs = []
for path, tag in CLEANED_FILES:
    if os.path.exists(path):
        df = pd.read_csv(path)
        new_docs = create_documents_from_df(df, tag)
        docs.extend(new_docs)
        print(f"✅ Added {len(new_docs)} documents from {path}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
print(f"✅ Split docs into {len(chunks)} chunks")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GEMINI_API_KEY)
vectorstore = FAISS.from_documents(chunks, embeddings)
os.makedirs("vectorstore", exist_ok=True)
vectorstore.save_local("vectorstore/samarth_vectorstore")
print("✅ Vector store saved: vectorstore/samarth_vectorstore/")
