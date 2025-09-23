from langchain.tools import BaseTool
from pydantic import Field

from langchain_ollama import ChatOllama

class RAGAgent(BaseTool):
    name: str = "RAGAgent"
    description: str = "Search for relevant information in the repository's README and generate a response based on the query and the found chunks"
    embedder: object = Field(default=None)
    vector_store: object = Field(default=None)
    llm: object = Field(default=None)

    def _run(self, query: str) -> str:
 
        query_embedding = self.embedder.embed_chunk(query)

        results = self.vector_store.query(query_embedding, top_k=3)

        context_items = []
        for r in results:
            meta = r.get("metadata", {})
            text = meta.get("text", "")
            title = meta.get("title", "")
            if title:
                context_items.append(f"Título: {title}\n{text}")
            else:
                context_items.append(text)
        context = "\n---\n".join(context_items)
       
        prompt = (
            "You are an expert assistant in GitHub repositories.\n"
            "Use the provided context to answer the user's question.\n"
            "If you don't find enough information, indicate it.\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        out = self.llm.invoke(prompt)  
        try:
            return out.content        
        except AttributeError:
            return str(out)  