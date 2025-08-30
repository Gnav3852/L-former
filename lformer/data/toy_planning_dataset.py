"""
Toy planning dataset for L-Former training
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Any
import random


class ToyPlanningDataset(Dataset):
    """Synthetic dataset requiring tool selection and planning"""
    
    def __init__(self, vocab_size: int = 1000, n_tools: int = 5, n_samples: int = 1000, max_seq_len: int = 50):
        self.vocab_size = vocab_size
        self.n_tools = n_tools
        self.n_samples = n_samples
        self.max_seq_len = max_seq_len
        
        # Special tokens
        self.bos_token = 0
        self.eos_token = 1
        self.pad_token = 2
        
        # Tool tokens
        self.tool_tokens = list(range(3, 3 + n_tools))
        
        # Generate synthetic examples
        self.examples = self._generate_examples()
    
    def _generate_examples(self) -> List[Dict[str, Any]]:
        """Generate synthetic planning examples"""
        examples = []
        
        # Tool selection examples
        tool_examples = self._generate_tool_examples()
        examples.extend(tool_examples)
        
        # Multi-step planning examples
        planning_examples = self._generate_planning_examples()
        examples.extend(planning_examples)
        
        # Value prediction examples
        value_examples = self._generate_value_examples()
        examples.extend(value_examples)
        
        # Shuffle examples
        random.shuffle(examples)
        
        return examples[:self.n_samples]
    
    def _generate_tool_examples(self) -> List[Dict[str, Any]]:
        """Generate examples requiring tool selection"""
        examples = []
        
        tool_descriptions = [
            "add numbers", "multiply numbers", "subtract numbers", 
            "divide numbers", "find maximum"
        ]
        
        for _ in range(self.n_samples // 3):
            # Random tool
            tool_idx = random.randint(0, self.n_tools - 1)
            
            # Generate input text
            if tool_idx == 0:  # add
                text = "I need to add 5 and 3 together"
                expected_tool = 0
            elif tool_idx == 1:  # multiply
                text = "What is 4 times 7?"
                expected_tool = 1
            elif tool_idx == 2:  # subtract
                text = "Calculate 10 minus 3"
                expected_tool = 2
            elif tool_idx == 3:  # divide
                text = "Divide 15 by 3"
                expected_tool = 3
            else:  # max
                text = "Find the largest number among 2, 8, 5"
                expected_tool = 4
            
            # Tokenize (simple character-level for toy dataset)
            input_ids = self._tokenize_text(text)
            
            examples.append({
                "input_ids": input_ids,
                "labels": input_ids,  # Same as input for LM
                "tool_labels": torch.tensor(expected_tool),
                "values": None,
                "plan_labels": None,
                "type": "tool"
            })
        
        return examples
    
    def _generate_planning_examples(self) -> List[Dict[str, Any]]:
        """Generate examples requiring multi-step planning"""
        examples = []
        
        plan_templates = [
            ("add then multiply", [0, 1]),  # tool indices
            ("multiply then add", [1, 0]),
            ("subtract then add", [2, 0]),
            ("add then subtract", [0, 2])
        ]
        
        for _ in range(self.n_samples // 3):
            plan_template = random.choice(plan_templates)
            plan_name, tool_sequence = plan_template
            
            # Generate input text
            text = f"I need to {plan_name} some numbers"
            
            # Tokenize
            input_ids = self._tokenize_text(text)
            
            # Create plan labels (sequence of tool tokens)
            plan_labels = [self.bos_token] + [self.tool_tokens[tool_idx] for tool_idx in tool_sequence] + [self.eos_token]
            plan_labels = torch.tensor(plan_labels)
            
            examples.append({
                "input_ids": input_ids,
                "labels": input_ids,
                "tool_labels": None,
                "values": None,
                "plan_labels": plan_labels,
                "type": "plan"
            })
        
        return examples
    
    def _generate_value_examples(self) -> List[Dict[str, Any]]:
        """Generate examples requiring value prediction"""
        examples = []
        
        for _ in range(self.n_samples // 3):
            # Random arithmetic problem
            a = random.randint(1, 20)
            b = random.randint(1, 20)
            operation = random.choice(["add", "multiply", "subtract"])
            
            if operation == "add":
                text = f"What is {a} plus {b}?"
                expected_value = a + b
            elif operation == "multiply":
                text = f"What is {a} times {b}?"
                expected_value = a * b
            else:  # subtract
                text = f"What is {a} minus {b}?"
                expected_value = a - b
            
            # Tokenize
            input_ids = self._tokenize_text(text)
            
            examples.append({
                "input_ids": input_ids,
                "labels": input_ids,
                "tool_labels": None,
                "values": torch.tensor([expected_value], dtype=torch.float),
                "plan_labels": None,
                "type": "value"
            })
        
        return examples
    
    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Simple character-level tokenization for toy dataset"""
        # Convert text to character indices
        chars = list(text.lower())
        
        # Map characters to token IDs (simple hash-based)
        token_ids = []
        for char in chars:
            if char.isalpha():
                token_id = (ord(char) - ord('a')) % (self.vocab_size - 10) + 10
            elif char.isdigit():
                token_id = (ord(char) - ord('0')) % 10 + 20
            elif char == ' ':
                token_id = 30
            else:
                token_id = 31
            
            token_ids.append(token_id)
        
        # Add BOS and EOS tokens
        token_ids = [self.bos_token] + token_ids + [self.eos_token]
        
        # Pad or truncate to max_seq_len
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        else:
            token_ids.extend([self.pad_token] * (self.max_seq_len - len(token_ids)))
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    # Stack tensors
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    # Handle optional fields
    tool_labels = None
    if any(item["tool_labels"] is not None for item in batch):
        tool_labels = torch.stack([item["tool_labels"] if item["tool_labels"] is not None else torch.tensor(0) for item in batch])
    
    values = None
    if any(item["values"] is not None for item in batch):
        values = torch.stack([item["values"] if item["values"] is not None else torch.tensor(0.0) for item in batch])
    
    plan_labels = None
    if any(item["plan_labels"] is not None for item in batch):
        # Find max plan length
        max_plan_len = max(len(item["plan_labels"]) if item["plan_labels"] is not None else 0 for item in batch)
        if max_plan_len > 0:
            plan_labels = torch.full((len(batch), max_plan_len), -100, dtype=torch.long)
            for i, item in enumerate(batch):
                if item["plan_labels"] is not None:
                    plan_len = len(item["plan_labels"])
                    plan_labels[i, :plan_len] = item["plan_labels"]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "tool_labels": tool_labels,
        "values": values,
        "plan_labels": plan_labels
    }


def create_dataloader(config, split: str = "train", batch_size: int = 32) -> DataLoader:
    """Create DataLoader for the dataset"""
    dataset = ToyPlanningDataset(
        vocab_size=config.vocab_size,
        n_tools=config.n_tools,
        n_samples=1000 if split == "train" else 200,
        max_seq_len=50
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn
    ) 