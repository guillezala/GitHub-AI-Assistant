from typing import List, Union
from transformers import AutoTokenizer


class Chunker:
    def __init__(self, max_tokens: int = 800, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_tokens = max_tokens

    def count_tokens(self, text: str) -> int:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return len(token_ids)

    def chunk(
        self,
        text: str,
        overlap: int = 0,
        return_metadata: bool = False
    ) -> Union[List[str], List[dict]]:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunks: List[Union[str, dict]] = []
        i = 0
        step = self.max_tokens - overlap if overlap else self.max_tokens

        if step <= 0:
            raise ValueError("overlap must be smaller than max_tokens and produce a positive step")

        while i < len(token_ids):
            chunk_ids = token_ids[i : i + self.max_tokens]
            # decode token ids back to string
            decoded = self.tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            if return_metadata:
                chunks.append({
                    "text": decoded,
                    "start": i,
                    "end": i + len(chunk_ids),
                })
            else:
                chunks.append(decoded)

            i += step

        return chunks
    
