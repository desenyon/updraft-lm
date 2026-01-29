import tiktoken

def get_tokenizer(name="gpt2"):
    try:
        tokenizer = tiktoken.get_encoding(name)
    except:
        tokenizer = tiktoken.get_encoding("gpt2")
    
    return tokenizer


class Tokenizer:
    def __init__(self, encoding_name="gpt2"):
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.vocab_size = self.encoding.n_vocab
    
    def encode(self, text):
        return self.encoding.encode(text, allowed_special=set())
    
    def decode(self, tokens):
        return self.encoding.decode(tokens)
    
    def __call__(self, text):
        return self.encode(text)
