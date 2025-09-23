from langchain.tools import BaseTool
from pydantic import Field

from langchain_ollama import ChatOllama

class RAGAgent(BaseTool):
    name: str = "RAGAgent"
    description: str = "Busca información relevante en el README del repositorio y genera una respuesta basada en la consulta y los chunks encontrados"
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
            "Eres un asistente experto en repositorios de GitHub.\n"
            "Utiliza el contexto proporcionado para responder la pregunta del usuario.\n"
            "Si no encuentras suficiente información, indícalo.\n"
            f"Contexto:\n{context}\n\nPregunta: {query}\nRespuesta:"
        )
        out = self.llm.invoke(prompt)  
        try:
            return out.content        
        except AttributeError:
            return str(out)  