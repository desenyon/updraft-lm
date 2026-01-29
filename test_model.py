import sys
import torch

def test_imports():
    print("Testing imports...")
    try:
        from model.gpt1 import GPT1Model
        from model.transformer import TransformerBlock, MultiHeadAttention
        from config import GPT1Config
        from data.tokenizer import Tokenizer
        from generator import Generator
        print("✓ Core imports successful")
        
        try:
            from utils import set_seed, get_device
            print("✓ Utils imports successful")
        except ImportError as e:
            print(f"⚠ Utils import warning: {e} (non-critical)")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_creation():
    print("\nTesting model creation...")
    try:
        from model.gpt1 import GPT1Model
        from config import GPT1Config
        
        config = GPT1Config()
        config.n_layers = 2
        config.n_heads = 4
        config.d_model = 256
        config.d_ff = 1024
        
        model = GPT1Model(config)
        params = model.get_num_params()
        print(f"✓ Model created with {params:,} parameters")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_forward_pass():
    print("\nTesting forward pass...")
    try:
        from model.gpt1 import GPT1Model
        from config import GPT1Config
        import torch
        
        config = GPT1Config()
        config.n_layers = 2
        config.n_heads = 4
        config.d_model = 256
        config.d_ff = 1024
        
        model = GPT1Model(config)
        model.eval()
        
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            logits, _ = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        print(f"✓ Forward pass successful, output shape: {logits.shape}")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def test_generation():
    print("\nTesting text generation...")
    try:
        from model.gpt1 import GPT1Model
        from config import GPT1Config
        from data.tokenizer import Tokenizer
        from generator import Generator
        import torch
        
        config = GPT1Config()
        config.n_layers = 2
        config.n_heads = 4
        config.d_model = 256
        config.d_ff = 1024
        config.max_seq_len = 128
        
        model = GPT1Model(config)
        tokenizer = Tokenizer()
        generator = Generator(model, tokenizer, config, 'cpu')
        
        prompt = "Hello world"
        output = generator.generate(prompt, max_length=10, temperature=1.0, top_k=50)
        
        print(f"✓ Generation successful")
        print(f"  Input: {prompt}")
        print(f"  Output length: {len(output[0])} chars")
        return True
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False

def test_tokenizer():
    print("\nTesting tokenizer...")
    try:
        from data.tokenizer import Tokenizer
        
        tokenizer = Tokenizer()
        
        text = "Hello, world! This is a test."
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        
        print(f"✓ Tokenizer working")
        print(f"  Original: {text}")
        print(f"  Tokens: {len(tokens)}")
        print(f"  Decoded: {decoded}")
        return True
    except Exception as e:
        print(f"✗ Tokenizer failed: {e}")
        return False

def test_attention():
    print("\nTesting attention mechanism...")
    try:
        from model.transformer import MultiHeadAttention
        import torch
        
        d_model = 256
        n_heads = 8
        batch_size = 2
        seq_len = 16
        
        attn = MultiHeadAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attn(x)
        
        assert output.shape == x.shape
        print(f"✓ Attention mechanism working, output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Attention test failed: {e}")
        return False

def main():
    print("="*80)
    print("Updraft-LM Test Suite")
    print("="*80)
    
    tests = [
        test_imports,
        test_model_creation,
        test_tokenizer,
        test_attention,
        test_forward_pass,
        test_generation,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    print("\n" + "="*80)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("="*80)
    
    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
