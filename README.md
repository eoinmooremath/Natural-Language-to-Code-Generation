# Natural Language to Code Generation

A comprehensive LoRA (Low-Rank Adaptation) fine-tuning system for converting natural language commands into structured API code. This project demonstrates synthetic data generation and model fine-tuning techniques using a simulated robot API as a case study.

## Overview

This project showcases modern ML engineering practices for code generation:

- **Synthetic Dataset Creation**: Async generation of 75,000+ training examples using OpenAI APIs
- **LoRA Fine-tuning**: Parameter-efficient training of CodeLlama models
- **Natural Language to Code**: Translation of human commands into structured API calls
- **Production Pipeline**: End-to-end training and evaluation system

**Note**: This uses a fictional robot API designed specifically for demonstrating code generation techniques. The API calls are syntactically valid Python but don't control actual hardware.

## Key Features

- **Large-Scale Synthetic Data**: Async pipeline generating diverse command-to-code pairs
- **Category-Weighted Sampling**: Optimized training across different command types
- **Interactive Demo**: Side-by-side comparison of base vs fine-tuned models
- **Production-Ready Training**: H100/A100 optimized with BF16 precision and gradient checkpointing

## Quick Start

### 1. Generate Training Data
```bash
python dataset_generator.py
```
Creates ~75,000 examples across movement, manipulation, navigation, and complex command categories.

### 2. Train the Model
```bash
python train_robot_lora_list.py --data-json robot_full_dataset.json --output-dir ./outputs/code-gen-lora
```

### 3. Run Interactive Demo
```bash
python robot_demo.py --lora ./outputs/code-gen-lora
```

## Example Transformations

```
Input: "go to the kitchen and pick up any red objects"
Output: 
robot.navigate.go_to_room(room="kitchen")
robot.sensors.detect(object_type="objects")
robot.gripper.pick_up(object_type="objects", color="red", size="any")
```

```
Input: "scan the area then move forward 3 meters"
Output:
robot.sensors.scan(range=10.0, angle=360)
robot.move_forward(distance=3.0, speed=1.0)
```

## Technical Highlights

### Synthetic Data Generation
- **Async Processing**: 15x speedup over sequential generation
- **Rate Limiting**: Respects OpenAI API constraints with exponential backoff
- **Category Diversity**: 12 command categories from simple movements to complex sequences
- **Quality Control**: LLM-based validation and deduplication

### Model Training
- **LoRA Configuration**: r=32, alpha=64 for efficient fine-tuning
- **Response-Only Loss**: Masks prompt tokens, trains only on code output
- **Dynamic Padding**: Optimized batching for variable-length sequences
- **Early Stopping**: Validation-based convergence with paired loss logging

## Simulated Robot API

The fictional API includes realistic robotics operations:

**Movement**: `move_forward()`, `turn()`, `move_to()`  
**Manipulation**: `gripper.pick_up()`, `gripper.drop()`  
**Sensors**: `sensors.scan()`, `sensors.detect()`  
**Navigation**: `navigate.go_to_room()`, `navigate.return_home()`

## Project Structure

```
├── dataset_generator.py      # Async synthetic data generation
├── train_robot_lora_list.py  # LoRA fine-tuning pipeline  
├── robot_demo.py            # Interactive Gradio demo
├── outputs/                 # Trained models and logs
└── examples/               # Sample datasets and configs
```

## Applications

This codebase demonstrates techniques applicable to:

- **Code generation** from natural language
- **API documentation** to code translation  
- **Domain-specific language** creation
- **Instruction following** for technical tasks
- **Large-scale synthetic data** creation

## Requirements

- Python 3.9+
- PyTorch with CUDA support
- Transformers, PEFT, Datasets libraries
- OpenAI API key (for data generation)

---

This project serves as a complete example of modern NLP engineering, from synthetic data creation through model deployment, using code generation as the target task.
