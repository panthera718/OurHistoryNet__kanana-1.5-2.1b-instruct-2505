from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================
# Paths
# ==========================
BASE_MODEL_PATH = "./kanana-1.5-2.1b-instruct-2505"
LORA_PATH       = "./Kanana-history-lora"   # output dir from 03_train_kanana_lora.py
DOCS_PATH       = "nh_raw.jsonl"            # output file from 01_crawl_nh_raw.py


# ==========================
# Document loading & search preparation
# ==========================
def load_docs(path=DOCS_PATH):
    docs = []
    texts = []

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"The file {path} does not exist. Please crawl it first.")

    with p.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = d.get("text", "").strip()
            if not text:
                continue
            docs.append(d)
            texts.append(text)

    print(f"[RAG] Loaded {len(docs)} docs from {path}")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2)
    )
    tfidf = vectorizer.fit_transform(texts)
    print("[RAG] Built TF-IDF matrix:", tfidf.shape)

    return docs, vectorizer, tfidf


def search_docs(query, docs, vectorizer, tfidf_matrix, k=3):
    """
    Returns the top k documents by TF-IDF cosine similarity for a user query.
    """
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    best_idx = scores.argsort()[::-1][:k]
    return [docs[i] for i in best_idx], [scores[i] for i in best_idx]


# ==========================
# Model loading (Base + optional LoRA)
# ==========================
def pick_dtype():
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(use_lora=True):
    dtype = pick_dtype()

    print("[Model] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"[Model] Loading base model (dtype={dtype}) ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=dtype,
        device_map="auto",
    )

    model = base_model
    if use_lora:
        print("[Model] Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, LORA_PATH)

    model.eval()
    return model, tokenizer



def get_input_device(model):
    """
    Return a reasonable device to place input tensors on.
    - If the model is loaded with device_map="auto", hf_device_map may exist.
    - Otherwise, fall back to model.device / first parameter device.
    """
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        # Pick the first non-cpu/disk device from the device map
        for dev in model.hf_device_map.values():
            if isinstance(dev, str) and dev not in ["cpu", "disk"]:
                return torch.device(dev)
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device

def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def build_inputs(tokenizer, messages):
    """
    Safer across transformers versions:
    - First build a prompt string using the model's built-in chat_template
    - Then tokenize with tokenizer(...)
    """
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(prompt_text, return_tensors="pt")


# ==========================
# Main Chat Loop
# ==========================
def main():
    # 1) RAG preparation
    docs, vectorizer, tfidf_matrix = load_docs(DOCS_PATH)

    # 2) Model/Tokenizer Loading
    model, tokenizer = load_model_and_tokenizer(use_lora=True)

    # 3) Reset conversation history (IMPORTANT: only use roles supported by the model chat_template)
    system_prompt = (
        "You are Kanana, a helpful bilingual (Korean/English) assistant.\n"
        "When answering Korean history questions, rely primarily on the provided reference text.\n"
        "If the answer is not in the references, say you do not know rather than guessing."
    )
    chat_history = [{"role": "system", "content": system_prompt}]

    print("Welcome to Kanana (History LoRA + RAG). Type 'exit' to exit.\n")

    while True:
        try:
            user_input = input("User: ")
        except EOFError:
            print("\nExit.")
            break

        if user_input.lower().strip() in ["exit", "quit"]:
            print("Exit.")
            break

        # --------------------------
        # 1) RAG retrieval
        # --------------------------
        retrieved_docs, scores = search_docs(
            user_input, docs, vectorizer, tfidf_matrix, k=3
        )

        context_chunks = []
        for d, s in zip(retrieved_docs, scores):
            # Use only the front part, not too long (adjust if necessary)
            snippet = d["text"][:5000]
            title = d.get("title", "")
            url = d.get("url", "")
            context_chunks.append(
                f"[Source Title] {title}\n[URL] {url}\n\n{snippet}"
            )

        context_text = "\n\n------------------------------\n\n".join(context_chunks)

        # --------------------------
        # 2) Build augmented user message
        # --------------------------
        augmented_user_input = (
            "다음은 국사편찬위원회 우리역사넷(신편 한국사)에서 가져온 참고 자료이다.\n"
            "아래 참고 자료와 사용자 질문을 바탕으로, 우리역사넷의 서술과 모순되지 않도록 신중하게 한국어로 답하라.\n"
            "가능하면 개념을 차근차근 설명하고, 여러 문단으로 나누어 상세하게 서술하라.\n"
            "참고 자료에 없는 내용은 추측하지 말고, 모른다고 대답해도 된다.\n\n"
            f"=== 참고 자료 시작 ===\n{context_text}\n=== 참고 자료 끝 ===\n\n"
            f"질문: {user_input}"
        )

        chat_history.append({"role": "user", "content": augmented_user_input})

        # --------------------------
        # 3) Tokenize & Generate
        # --------------------------
        inputs = build_inputs(tokenizer, chat_history)
        device = get_input_device(model)
        inputs = move_batch_to_device(inputs, device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                top_p=0.9,
                temperature=0.8,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Cut out the prompt part from the entire output and take only the newly generated part.
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        ai_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        print(f"Kanana: {ai_response}\n")

        # Add assistant response
        chat_history.append({"role": "assistant", "content": ai_response})


if __name__ == "__main__":
    main()
