import torch
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class AMDRAGSystem:
    def __init__(self, llm_model="meta-llama/Llama-3-8b-hf", embed_model="BAAI/bge-large-en-v1.5"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing RAG on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        # Embedding model on AMD GPU
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

        # LLM with ROCm
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
        model = AutoModelForCausalLM.from_pretrained(llm_model, torch_dtype=torch.bfloat16, device_map="auto")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
        self.llm = HuggingFacePipeline(pipeline=pipe)

        self.vectordb = None
        self.qa_chain = None

    def ingest_documents(self, documents, collection_name="default"):
        self.vectordb = Chroma.from_texts(
            documents, self.embeddings, collection_name=collection_name
        )
        retriever = self.vectordb.as_retriever(search_kwargs={"k": 5})
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm, chain_type="stuff", retriever=retriever
        )
        print(f"Ingested {len(documents)} documents")

    def query(self, question):
        if not self.qa_chain:
            raise ValueError("No documents ingested yet")
        return self.qa_chain.run(question)

if __name__ == "__main__":
    rag = AMDRAGSystem()
    docs = ["AMD Instinct MI300X is the world's most advanced accelerator for AI...",
            "ROCm is AMD's open-source software platform for GPU computing..."]
    rag.ingest_documents(docs)
    print(rag.query("What is MI300X?"))
