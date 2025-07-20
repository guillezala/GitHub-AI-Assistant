import tiktoken
from typing import List, Union

class Chunker:
    def __init__(self, max_tokens: int = 800):
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def chunk(
        self,
        text: str,
        overlap: int = 0,
        return_metadata: bool = False
    ) -> Union[List[str], List[dict]]:
        tokens = self.encoder.encode(text)
        chunks = []
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i : i + self.max_tokens]
            decoded = self.encoder.decode(chunk_tokens)

            if return_metadata:
                chunks.append({
                    "text": decoded,
                    "start": i,
                    "end": i + len(chunk_tokens)
                })
            else:
                chunks.append(decoded)

            i += self.max_tokens - overlap if overlap else self.max_tokens
        return chunks
    

from transformers import AutoTokenizer

class LlamaChunker:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", max_tokens=800):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens

    def chunk(self, text, overlap=0, return_metadata=False):
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        i = 0
        while i < len(input_ids):
            chunk_ids = input_ids[i: i + self.max_tokens]
            decoded = self.tokenizer.decode(chunk_ids)

            if return_metadata:
                chunks.append({
                    "text": decoded,
                    "start": i,
                    "end": i + len(chunk_ids)
                })
            else:
                chunks.append(decoded)

            i += self.max_tokens - overlap if overlap else self.max_tokens
        return chunks