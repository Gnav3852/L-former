import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import random
from transformers import GPT2Tokenizer


class RealTextDataset(Dataset):
    """Dataset with REAL TEXT for tool selection and planning"""
    
    def __init__(self, n_tools: int = 3, n_samples: int = 1000, max_seq_len: int = 100):
        self.n_tools = n_tools
        self.n_samples = n_samples
        self.max_seq_len = max_seq_len
        
        # Real tokenizer!
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Generate examples with REAL TEXT
        self.examples = self._generate_examples()
    
    def _generate_examples(self) -> List[Dict[str, Any]]:
        """Generate examples with meaningful text"""
        examples = []
        
        # Tool selection examples
        tool_examples = self._generate_tool_examples()
        examples.extend(tool_examples)
        
        # Planning examples
        planning_examples = self._generate_planning_examples()
        examples.extend(planning_examples)
        
        # Value examples
        value_examples = self._generate_value_examples()
        examples.extend(value_examples)
        
        random.shuffle(examples)
        return examples[:self.n_samples]
    
    def _generate_tool_examples(self) -> List[Dict[str, Any]]:
        """Tool selection with REAL TEXT - BALANCED VERSION"""
        examples = []
        
        # Clear, distinct text for each tool
        tool_texts = [
            "I need to add 5 and 3 together",           # Tool 0: Add
            "What is 4 times 7?",                       # Tool 1: Multiply  
            "Calculate 10 minus 3"                      # Tool 2: Subtract
        ]
        
        # Generate EXACTLY equal numbers for each tool
        samples_per_tool = self.n_samples // 3
        for tool_idx in range(self.n_tools):
            for _ in range(samples_per_tool):
                text = tool_texts[tool_idx]
                
                # Tokenize with real tokenizer
                input_ids = self.tokenizer.encode(text, max_length=self.max_seq_len, truncation=True, padding='max_length')
                input_ids = torch.tensor(input_ids)
                
                examples.append({
                    "input_ids": input_ids,
                    "labels": input_ids,
                    "tool_labels": torch.tensor(tool_idx),
                    "values": None,
                    "plan_labels": None,
                    "type": "tool"
                })
        
        return examples
    
    def _generate_planning_examples(self) -> List[Dict[str, Any]]:
        """Planning with REAL TEXT"""
        examples = []
        
        plan_templates = [
            ("add then multiply", [0, 1], "I need to add 5 and 3, then multiply the result by 2"),
            ("multiply then add", [1, 0], "What is 4 times 7, then add 10 to the result"),
            ("subtract then add", [2, 0], "Calculate 10 minus 3, then add 5 to the result"),
            ("add then subtract", [0, 2], "I need to add 8 and 4, then subtract 3 from the total")
        ]
        
        for _ in range(self.n_samples // 3):
            plan_name, tool_sequence, text = random.choice(plan_templates)
            
            input_ids = self.tokenizer.encode(text, max_length=self.max_seq_len, truncation=True, padding='max_length')
            input_ids = torch.tensor(input_ids)
            
            # Create plan labels
            plan_labels = [self.tokenizer.bos_token_id] + [self.tokenizer.eos_token_id + 1 + i for i in tool_sequence] + [self.tokenizer.eos_token_id]
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
        """Value prediction with REAL TEXT"""
        examples = []
        
        for _ in range(self.n_samples // 3):
            a = random.randint(1, 20)
            b = random.randint(1, 20)
            operation = random.choice(["add", "multiply", "subtract"])
            
            if operation == "add":
                text = f"What is {a} plus {b}?"
                expected_value = a + b
            elif operation == "multiply":
                text = f"What is {a} times {b}?"
                expected_value = a * b
            else:
                text = f"What is {a} minus {b}?"
                expected_value = a - b
            
            input_ids = self.tokenizer.encode(text, max_length=self.max_seq_len, truncation=True, padding='max_length')
            input_ids = torch.tensor(input_ids)
            
            examples.append({
                "input_ids": input_ids,
                "labels": input_ids,
                "tool_labels": None,
                "values": torch.tensor([expected_value], dtype=torch.float),
                "plan_labels": None,
                "type": "value"
            })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    def print_stats(self):
        """Print dataset statistics to verify balance"""
        tool_counts = [0] * self.n_tools
        type_counts = {"tool": 0, "plan": 0, "value": 0}
        
        for example in self.examples:
            if example["type"] == "tool" and example["tool_labels"] is not None:
                tool_counts[example["tool_labels"].item()] += 1
            type_counts[example["type"]] += 1
        
        print(f"Dataset Statistics:")
        print(f"  Total examples: {len(self.examples)}")
        print(f"  Tool distribution: {tool_counts}")
        print(f"  Type distribution: {type_counts}")
        
        # Verify balance
        if len(set(tool_counts)) == 1:
            print("  ✅ Tool distribution is perfectly balanced!")
        else:
            print("  ❌ Tool distribution is imbalanced!")
            print(f"  Expected: {self.n_samples // 3} per tool")
            print(f"  Actual: {tool_counts}")


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    # Handle optional fields
    tool_labels = None
    if any(item["tool_labels"] is not None for item in batch):
        tool_labels_list = []
        for item in batch:
            if item["tool_labels"] is not None:
                tool_labels_list.append(item["tool_labels"])
            else:
                if tool_labels_list:
                    dummy_shape = tool_labels_list[0].shape
                    tool_labels_list.append(torch.zeros(dummy_shape, dtype=torch.long))
                else:
                    tool_labels_list.append(torch.tensor(0, dtype=torch.long))
        tool_labels = torch.stack(tool_labels_list)
    
    values = None
    if any(item["values"] is not None for item in batch):
        values_list = []
        for item in batch:
            if item["values"] is not None:
                values_list.append(item["values"])
            else:
                if values_list:
                    dummy_shape = values_list[0].shape
                    values_list.append(torch.zeros(dummy_shape, dtype=torch.float))
                else:
                    values_list.append(torch.tensor([0.0], dtype=torch.float))
        values = torch.stack(values_list)
    
    plan_labels = None
    if any(item["plan_labels"] is not None for item in batch):
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
    """Create DataLoader for the real text dataset"""
    dataset = RealTextDataset(
        n_tools=config.n_tools,
        n_samples=1000 if split == "train" else 200,
        max_seq_len=100
    )
    
    # Print dataset statistics for verification
    if split == "train":
        dataset.print_stats()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn
    )