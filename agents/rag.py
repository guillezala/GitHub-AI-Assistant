from langchain.tools import BaseTool
from pydantic import Field

class RAGAgent(BaseTool):
    name: str = "RAGAgent"
    description: str = "Busca información relevante en Pinecone y genera una respuesta basada en la consulta y los chunks encontrados"
    embedder: object = Field(...)
    vector_store: object = Field(...)
    llm: object = Field(...)

    def _run(self, query: str) -> str:
        # 1. Obtener embedding de la consulta
        query_embedding = self.embedder.embed_chunk(query)
        # 2. Buscar chunks relevantes en Pinecone
        results = self.vector_store.query(query_embedding, top_k=5)
        # 3. Extraer textos y metadatos de los chunks encontrados
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
        # 4. Generar respuesta usando el LLM y el contexto
        prompt = (
            "Eres un asistente experto en repositorios de GitHub.\n"
            "Utiliza el contexto proporcionado para responder la pregunta del usuario.\n"
            "Si no encuentras suficiente información, indícalo.\n"
            f"Contexto:\n{context}\n\nPregunta: {query}\nRespuesta:"
        )
        respuesta_obj = self.llm.generate([prompt])
        respuesta = respuesta_obj.generations[0][0].text
        return respuesta
