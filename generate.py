
from model import load_tokenizer_and_model
from config import device
from data_utils import extract_corpus, load_queries
from preprocess import prepare_retrieval_sources
import torch

tokenizer, model = load_tokenizer_and_model()
corpus = extract_corpus()
sources = prepare_retrieval_sources(corpus)
queries = load_queries()

prompt = """
You are an expert assistant. Using the provided context, provide a clear and explanatory completion to the answer based strictly on the information in the context. Avoid conversational or chat-like responses.

Context: "{context}"

Query: "{query}"

Answer:"""


# custom_prompts = [prompt.format(context = source[0][0].page_content, query = query) for source, query in zip(sources, queries)]
custom_prompts = [prompt.format(context = source.page_content, query = query) for source, query in zip(sources, queries)]
inputs = tokenizer(custom_prompts, return_tensors="pt", padding = True).to(device)


def generate_batch(prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding = True).to(device)

    torch.cuda.empty_cache()
    
    outputs = model.generate(
        **inputs,
        max_length=1900,  # Maximum length of the output
        temperature=0.5,  # Controls randomness (lower is more focused, higher is creative)
        top_p=0.9,       # Nucleus sampling
        do_sample=True,  # Enables sampling for diverse output
        repetition_penalty = 1.2,
        num_return_sequences=1  # Number of generated sequences
    ) 
    
    gen_texts = tokenizer.batch_decode(outputs, skip_special_tokens = True)
    print("batch completed")
    
    return gen_texts


def write_sols():
    bs = 3
    generated_texts = []
    
    for i in range(0, len(custom_prompts), bs):
        start = i; end = min(len(custom_prompts), i + bs)
        batch = custom_prompts[start : end]
        generated_texts.extend(generate_batch(batch))
    
    file = "answers.txt"

    with open(file, 'w') as f:
        ct = 1
        for query, text in zip(queries, generated_texts):
            sol = text.split('Answer')[-1][1:]
            f.write(f"[{ct}] {query}\n \t{sol}\n\n")
            ct+=1


if __name__ == "__main__":
    write_sols()
