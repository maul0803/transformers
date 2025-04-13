import os
import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs

#RuntimeError: This event loop is already running
import nest_asyncio
nest_asyncio.apply()

# === Setup logging ===
logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

WORKING_DIR = "./nano_graphrag_cache_flanT5"
cache_dir = "/mnt/lustre/scratch/nlsas/home/ulc/cursos/curso341/transformers/GRAPH_RAG2/cache"

# === Load embedding model ===
EMBED_MODEL = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2", cache_folder=WORKING_DIR, device="cpu"
)

@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)

# === Configuration ===
llm_model_name = "google/flan-t5-base"
#llm_model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
#llm_model_name = "facebook/bart-large"
#llm_model_name = "EleutherAI/gpt-j-6B"

#llm_tokenizer_name = "Qwen/Qwen2.5-7B"
#llm_model_name = "Qwen/Qwen2.5-7B"

llm_tokenizer_name = "INSAIT-Institute/BgGPT-Gemma-2-27B-IT-v1.0"
llm_model_name = "INSAIT-Institute/BgGPT-Gemma-2-27B-IT-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print('DEVICE:', DEVICE)

# === Load the Seq2Seq LLM ===
tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_name, cache_dir = cache_dir)
#llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name, cache_dir = cache_dir)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, cache_dir = cache_dir)

# test of the model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
llm_model = llm_model.to(DEVICE)

prompt = "Can you explain gravity in simple terms?"

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(DEVICE)

with torch.no_grad():
    output = llm_model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        num_beams=1
    )

result = tokenizer.decode(output[0], skip_special_tokens=True)
print("LLM Response:", result)

async def my_llm_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    input_text = prompt if not system_prompt else f"{system_prompt}\n{prompt}"
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        output_tokens = llm_model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
        )
    output_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print("🧪 LLM output:", output_text)
    return output_text

#prompt = "Please extract the named entities and their types (Person, Location, Date, etc.) from the following text: 'Albert Einstein was born in Ulm, Germany, in 1879. He is known for developing the theory of relativity.' Return the result in the following format:[ {'entity': 'Albert Einstein', 'type': 'Person'}, {'entity': 'Ulm', 'type': 'Location'}, {'entity': 'Germany', 'type': 'Location'}, {'entity': '1879', 'type': 'Date'} ]"
#await my_llm_complete(prompt)

# === Utility ===
def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)

def get_rag(enable_cache=False):
    return GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=enable_cache,
        best_model_func=my_llm_complete,
        cheap_model_func=my_llm_complete,
        embedding_func=local_embedding,
    )

# === Main functions ===
def query(question="What is X"):
    rag = get_rag()
    print(
        rag.query(
            question, param=QueryParam(mode="global")
        )
    )

def insert():
    from time import time

    with open("./dataset/test.txt", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()

    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    rag = get_rag()
    start = time()
    rag.insert(FAKE_TEXT)
    print("indexing time:", time() - start)
    return FAKE_TEXT

logging.getLogger("nano-graphrag").setLevel(logging.DEBUG)
# === Entry point ===
if __name__ == "__main__":
    FAKE_TEXT = insert()
    query()

print(FAKE_TEXT)