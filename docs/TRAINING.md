# Training Guide for Robot-LoRA

This guide covers everything you need to know about training LoRA models for natural language robot programming.

## 🎯 Training Overview

Robot-LoRA uses Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA) to efficiently train models for converting natural language commands into robot control code.

### Key Benefits of LoRA
- **Memory Efficient**: Trains only ~1% of model parameters
- **Fast Training**: Significantly faster than full fine-tuning
- **Preserves Base Model**: Original model knowledge is retained
- **Easy Deployment**: LoRA adapters are small and portable

## 📊 Dataset Requirements

### Dataset Format
The training system expects JSON files with the following structure:

```json
[
  {
    "input": "move forward 3 meters",
    "output": "robot.move_forward(distance=3.0, speed=1.0)",
    "category": "movement"
  },
  {
    "input": "pick up the red ball",
    "output": "robot.gripper.pick_up(object_type=\"ball\", color=\"red\", size=\"medium\")",
    "category": "manipulation"
  }
]
```

### Required Fields
- **input**: Natural language command
- **output**: Corresponding robot API call(s)
- **category**: Command category (optional but recommended)

### Dataset Size Recommendations
- **Minimum**: 1,000 examples for basic functionality
- **Small**: 5,000 examples for decent performance
- **Medium**: 20,000 examples for good performance
- **Large**: 50,000+ examples for excellent performance

## ⚙️ Training Configuration

### Basic Training Command
```bash
python train_robot.py \
    --data-json robot_dataset.json \
    --output-dir ./models/robot-lora \
    --num-epochs 3 \
    --batch-size 4 \
    --learning-rate 1e-4
```

### Separate Train/Validation Files
```bash
python train_robot.py \
    --train-json robot_train.json \
    --val-json robot_val.json \
    --output-dir ./models/robot-lora
```

### Key Parameters

#### Model Configuration
- `--model-name`: Base model to fine-tune (default: CodeLlama-7b-Python-hf)
- `--max-length`: Maximum sequence length (default: 256)
- `--lora-r`: LoRA rank (default: 32, higher = more parameters)
- `--lora-alpha`: LoRA scaling factor (default: 64)

#### Training Parameters
- `--num-epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: 4)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--warmup-ratio`: Learning rate warmup (default: 0.1)

#### Monitoring
- `--eval-steps`: Steps between evaluations (default: 100)
- `--save-steps`: Steps between checkpoints (default: 500)
- `--logging-steps`: Steps between logs (default: 50)

## 🖥️ Hardware Requirements

### GPU Memory Requirements
| Model Size | Minimum VRAM | Recommended VRAM | Batch Size |
|------------|--------------|------------------|------------|
| 7B         | 16 GB        | 24 GB            | 4-8        |
| 13B        | 24 GB        | 40 GB            | 2-4        |
| 34B        | 48 GB        | 80 GB            | 1-2        |

### Optimization Tips for Limited Memory
```bash
# Reduce batch size
python train_robot.py --batch-size 2

# Use gradient checkpointing
python train_robot.py --gradient-checkpointing

# Use smaller model
python train_robot.py --model-name microsoft/DialoGPT-small
```

## 📈 Training Process

### 1. Data Preparation
```python
# Load and validate dataset
with open('robot_dataset.json', 'r') as f:
    data = json.load(f)

# Check data quality
for item in data[:5]:
    print(f"Input: {item['input']}")
    print(f"Output: {item['output']}")
    print("---")
```

### 2. Model Initialization
The training script automatically:
- Loads the base model (CodeLlama by default)
- Applies LoRA configuration
- Sets up response-only loss (ignores prompt tokens)
- Configures BF16 precision for efficiency

### 3. Training Loop
```
Epoch 1/3
├── Batch 1-100: Training loss decreases from ~2.5 to ~1.8
├── Evaluation: Validation loss ~1.5
├── Batch 101-200: Training loss ~1.2
└── Checkpoint saved

Epoch 2/3
├── Batch 201-300: Training loss ~0.8
├── Evaluation: Validation loss ~0.6
└── ...

Final Model: Training loss ~0.2, Validation loss ~0.3
```

## 📊 Monitoring Training

### Key Metrics to Watch
1. **Training Loss**: Should decrease consistently
2. **Validation Loss**: Should track training loss without diverging
3. **Learning Rate**: Should follow warmup schedule
4. **GPU Memory**: Should remain stable

### Loss Patterns
- **Good**: Smooth decrease in both training and validation loss
- **Overfitting**: Training loss much lower than validation loss
- **Underfitting**: Both losses plateau at high values
- **Instability**: Erratic loss fluctuations

### Example Training Logs
```
Step 100: train_loss=1.234, eval_loss=1.456, lr=5e-5
Step 200: train_loss=0.987, eval_loss=1.123, lr=1e-4
Step 300: train_loss=0.678, eval_loss=0.834, lr=1e-4
```

## 🎛️ Advanced Configuration

### Custom LoRA Settings
```python
# High-rank LoRA for better performance
python train_robot.py --lora-r 64 --lora-alpha 128

# Target more modules
python train_robot.py --lora-target-modules "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj"
```

### Learning Rate Scheduling
```python
# Cosine annealing
python train_robot.py --lr-scheduler-type cosine

# Linear warmup + decay
python train_robot.py --warmup-ratio 0.1 --lr-scheduler-type linear
```

### Data Loading Optimization
```python
# Group sequences by length for efficiency
python train_robot.py --group-by-length

# Increase data loading workers
python train_robot.py --dataloader-num-workers 8
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce `--batch-size`
- Enable `--gradient-checkpointing`
- Use smaller model
- Reduce `--max-length`

#### 2. Poor Performance
**Symptoms:** High validation loss, poor generation quality
**Solutions:**
- Increase dataset size
- Train for more epochs
- Adjust learning rate
- Check data quality

#### 3. Overfitting
**Symptoms:** Training loss << Validation loss
**Solutions:**
- Reduce learning rate
- Add more validation data
- Use early stopping
- Increase regularization

#### 4. Training Stalls
**Symptoms:** Loss stops decreasing
**Solutions:**
- Increase learning rate
- Check for data repetition
- Verify gradient flow
- Try different optimizer

### Debugging Commands
```bash
# Check GPU memory usage
nvidia-smi

# Monitor training in real-time
watch -n 1 'tail -20 training.log'

# Test model during training
python robot_demo.py --lora-path ./models/robot-lora/checkpoint-500
```

## 📋 Training Checklist

### Pre-Training
- [ ] Dataset prepared and validated
- [ ] GPU memory sufficient for batch size
- [ ] Output directory exists and is writable
- [ ] Base model downloaded and accessible

### During Training
- [ ] Monitor training and validation loss
- [ ] Check GPU utilization and memory
- [ ] Verify checkpoints are being saved
- [ ] Watch for signs of overfitting

### Post-Training
- [ ] Final model saved successfully
- [ ] Test generation quality
- [ ] Compare with validation metrics
- [ ] Backup trained model

## 🚀 Advanced Training Strategies

### Multi-Stage Training
```bash
# Stage 1: Basic commands (lower LR)
python train_robot.py --data-json basic_commands.json --learning-rate 5e-5 --num-epochs 2

# Stage 2: Complex commands (higher LR)
python train_robot.py --data-json complex_commands.json --learning-rate 1e-4 --num-epochs 3
```

### Curriculum Learning
```bash
# Start with simple examples, gradually add complexity
python train_robot.py --data-json simple_dataset.json --num-epochs 2
python train_robot.py --data-json medium_dataset.json --num-epochs 2 --resume-from ./checkpoint-1000
python train_robot.py --data-json complex_dataset.json --num-epochs 2 --resume-from ./checkpoint-2000
```

### Ensemble Training
```bash
# Train multiple models with different configurations
python train_robot.py --lora-r 32 --output-dir ./models/model1
python train_robot.py --lora-r 64 --output-dir ./models/model2
python train_robot.py --lora-alpha 32 --output-dir ./models/model3
```

## 📊 Performance Optimization

### Training Speed
- Use `--bf16` for faster training on modern GPUs
- Enable `--dataloader-pin-memory` for faster data loading
- Use `--group-by-length` to reduce padding
- Increase `--dataloader-num-workers`

### Memory Optimization
- Enable `--gradient-checkpointing`
- Use smaller `--max-length`
- Reduce `--batch-size`
- Use `--optim adamw_torch_fused` for memory efficiency

### Quality Optimization
- Increase dataset size and diversity
- Use higher LoRA rank (`--lora-r 64`)
- Train for more epochs
- Tune learning rate carefully

## 🎯 Expected Results

### Training Metrics
- **Good Training**: Final loss < 0.5
- **Excellent Training**: Final loss < 0.2
- **Validation Gap**: < 0.3 difference from training loss

### Generation Quality
- **Basic**: Correct API syntax, simple commands
- **Good**: Handles complex commands, proper parameters
- **Excellent**: Contextual understanding, multi-step sequences

For more details, see the [API documentation](API.md) and [examples](../examples/).
