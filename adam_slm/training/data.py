"""
Data loading utilities for ADAM SLM training
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Union
import tiktoken
import numpy as np


class AdamDataset(Dataset):
    """
    Dataset class for ADAM SLM training
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 1024,
        stride: Optional[int] = None,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length
        
        # Tokenize all texts
        self.examples = self._prepare_examples()
        
    def _prepare_examples(self) -> List[Dict[str, torch.Tensor]]:
        """Prepare training examples from texts"""
        examples = []
        
        for text in self.texts:
            # Tokenize text
            token_ids = self.tokenizer.encode(text)
            
            # Create sliding windows
            for i in range(0, len(token_ids) - self.max_length + 1, self.stride):
                input_ids = token_ids[i:i + self.max_length]
                
                if len(input_ids) == self.max_length:
                    examples.append({
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(input_ids, dtype=torch.long),
                    })
                    
        return examples
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


class AdamDataLoader:
    """
    Advanced data loader for ADAM SLM with various utilities
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn,
        )
        
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching"""
        # Stack tensors
        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        
        # Create attention mask (all ones for causal LM)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        
    def __iter__(self):
        return iter(self.dataloader)
        
    def __len__(self):
        return len(self.dataloader)


def create_dataloader(
    texts: Union[List[str], str],
    tokenizer,
    max_length: int = 1024,
    batch_size: int = 32,
    stride: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    train_test_split: Optional[float] = None,
) -> Union[AdamDataLoader, tuple]:
    """
    Create data loader(s) from texts
    
    Args:
        texts: List of texts or single text
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        batch_size: Batch size
        stride: Stride for sliding window (defaults to max_length)
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
        train_test_split: If provided, split data into train/test
        
    Returns:
        Single dataloader or tuple of (train_loader, test_loader)
    """
    if isinstance(texts, str):
        texts = [texts]
        
    # Create dataset
    dataset = AdamDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
    )
    
    if train_test_split is not None:
        # Split dataset
        total_size = len(dataset)
        train_size = int(total_size * (1 - train_test_split))
        test_size = total_size - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )
        
        # Create data loaders
        train_loader = AdamDataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        
        test_loader = AdamDataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle test data
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,  # Don't drop last for evaluation
        )
        
        return train_loader, test_loader
    else:
        # Single data loader
        return AdamDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )


def load_text_data(file_path: str, encoding: str = "utf-8") -> str:
    """Load text data from file"""
    with open(file_path, 'r', encoding=encoding) as f:
        return f.read()


def prepare_shakespeare_data() -> str:
    """
    Prepare Shakespeare dataset for training
    Downloads if not available locally
    """
    try:
        # Try to load local file first
        with open("shakespeare.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        # Download from internet
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, "shakespeare.txt")
        with open("shakespeare.txt", "r", encoding="utf-8") as f:
            text = f.read()
            
    return text


def get_tokenizer(tokenizer_name: str = "gpt2"):
    """Get tokenizer by name"""
    if tokenizer_name == "gpt2":
        return tiktoken.get_encoding("gpt2")
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer_name}")


def compute_dataset_stats(dataset: AdamDataset) -> Dict[str, Any]:
    """Compute statistics about the dataset"""
    total_tokens = sum(len(example["input_ids"]) for example in dataset.examples)
    
    return {
        "num_examples": len(dataset),
        "total_tokens": total_tokens,
        "avg_tokens_per_example": total_tokens / len(dataset) if len(dataset) > 0 else 0,
        "max_length": dataset.max_length,
        "stride": dataset.stride,
    }
