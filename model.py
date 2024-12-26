
from transformers import GemmaTokenizerFast, AutoModelForCausalLM, BitsAndBytesConfig
from config import device

def load_tokenizer_and_model():

    model_ckpt = '/kaggle/input/gemma/transformers/2b-it/3'
        
    tokenizer = GemmaTokenizerFast.from_pretrained(model_ckpt)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_ckpt,
        device_map = "auto"
    ).to(device)
    
    return tokenizer, model
