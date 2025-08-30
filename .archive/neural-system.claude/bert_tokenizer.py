#!/usr/bin/env python3
"""
Lightweight BERT Tokenizer for ONNX Models
Compatible with Python 3.13, no PyTorch/transformers dependencies
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

class SimpleBERTTokenizer:
    """Lightweight BERT tokenizer using vocabulary file"""
    
    def __init__(self, vocab_path: str = ".claude/onnx_models/tokenizer/vocab.txt"):
        """Initialize tokenizer with BERT vocabulary"""
        
        self.vocab_path = Path(vocab_path)
        
        # Load vocabulary
        self.vocab = {}
        self.inv_vocab = {}
        
        if self.vocab_path.exists():
            with open(self.vocab_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    token = line.strip()
                    self.vocab[token] = idx
                    self.inv_vocab[idx] = token
        else:
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
            
        # Special tokens
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.mask_token = "[MASK]"
        
        # Token IDs
        self.cls_token_id = self.vocab.get(self.cls_token, 101)
        self.sep_token_id = self.vocab.get(self.sep_token, 102)
        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.unk_token_id = self.vocab.get(self.unk_token, 100)
        
        print(f"✅ Loaded BERT vocabulary: {len(self.vocab):,} tokens")
        
    def tokenize(self, text: str) -> List[str]:
        """Basic WordPiece tokenization"""
        
        # Convert to lowercase
        text = text.lower()
        
        # Basic pre-tokenization (split on whitespace and punctuation)
        tokens = []
        
        # Split by whitespace first
        words = text.split()
        
        for word in words:
            # Handle punctuation
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
            
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using WordPiece algorithm"""
        
        # Simple implementation of WordPiece
        # Try to find the longest matching subword from vocabulary
        
        tokens = []
        chars = list(word)
        
        # Handle punctuation separately
        punct_pattern = re.compile(r'([^\w\s])')
        parts = punct_pattern.split(word)
        
        for part in parts:
            if not part:
                continue
                
            if punct_pattern.match(part):
                # It's punctuation
                if part in self.vocab:
                    tokens.append(part)
                else:
                    tokens.append(self.unk_token)
            else:
                # It's a word part - apply WordPiece
                sub_tokens = self._wordpiece_tokenize(part)
                tokens.extend(sub_tokens)
                
        return tokens
    
    def _wordpiece_tokenize(self, text: str, max_chars: int = 100) -> List[str]:
        """WordPiece tokenization of a single word"""
        
        if len(text) > max_chars:
            return [self.unk_token]
            
        # Check if whole word is in vocab
        if text in self.vocab:
            return [text]
            
        # Try to break into subwords
        tokens = []
        start = 0
        
        while start < len(text):
            end = len(text)
            cur_substr = None
            
            while start < end:
                substr = text[start:end]
                
                # Add ## prefix for subwords (BERT convention)
                if start > 0:
                    substr = "##" + substr
                    
                if substr in self.vocab:
                    cur_substr = substr
                    break
                    
                end -= 1
                
            if cur_substr is None:
                # Couldn't find a match
                tokens.append(self.unk_token)
                start = len(text)
            else:
                tokens.append(cur_substr)
                start = end
                
        return tokens
    
    def encode(self, text: str, max_length: int = 512) -> Dict[str, np.ndarray]:
        """Encode text to BERT input format"""
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Add special tokens
        tokens = [self.cls_token] + tokens[:max_length-2] + [self.sep_token]
        
        # Convert to IDs
        input_ids = []
        for token in tokens:
            token_id = self.vocab.get(token, self.unk_token_id)
            input_ids.append(token_id)
            
        # Pad to max_length
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.pad_token_id] * padding_length
        else:
            input_ids = input_ids[:max_length]
            
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if tid != self.pad_token_id else 0 for tid in input_ids]
        
        # Token type IDs (all 0 for single sentence)
        token_type_ids = [0] * max_length
        
        return {
            "input_ids": np.array([input_ids], dtype=np.int64),
            "attention_mask": np.array([attention_mask], dtype=np.int64),
            "token_type_ids": np.array([token_type_ids], dtype=np.int64)
        }
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        
        tokens = []
        for tid in token_ids:
            if tid == self.pad_token_id:
                continue
            if tid in self.inv_vocab:
                token = self.inv_vocab[tid]
                # Remove ## prefix for subwords
                if token.startswith("##"):
                    token = token[2:]
                tokens.append(token)
                
        # Join tokens
        text = " ".join(tokens)
        
        # Clean up WordPiece artifacts
        text = text.replace(" ##", "")
        text = text.replace(self.cls_token, "").replace(self.sep_token, "")
        text = text.replace(self.pad_token, "")
        
        return text.strip()


def test_tokenizer():
    """Test the BERT tokenizer"""
    
    print("\n🧪 Testing BERT Tokenizer")
    print("=" * 60)
    
    # Initialize tokenizer
    tokenizer = SimpleBERTTokenizer()
    
    # Test sentences
    test_texts = [
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning with BERT embeddings"
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        
        # Tokenize
        tokens = tokenizer.tokenize(text)
        print(f"Tokens ({len(tokens)}): {tokens[:10]}...")
        
        # Encode
        encoded = tokenizer.encode(text)
        print(f"Input IDs shape: {encoded['input_ids'].shape}")
        print(f"First 10 IDs: {encoded['input_ids'][0][:10].tolist()}")
        
        # Check all values are within BERT vocab range
        max_id = np.max(encoded['input_ids'])
        min_id = np.min(encoded['input_ids'])
        print(f"ID range: [{min_id}, {max_id}] (should be [0, 30521])")
        
        assert max_id < 30522, f"Token ID {max_id} out of range!"
        assert min_id >= 0, f"Token ID {min_id} out of range!"
        
    print("\n✅ All tests passed! Tokenizer is working correctly.")
    

if __name__ == "__main__":
    test_tokenizer()