# Robot-LoRA Usage Examples

This document provides comprehensive examples of how to use the Robot-LoRA system for natural language robot programming.

## 🚀 Quick Start Examples

### Basic Movement Commands

```python
# Natural language input → Robot code output

Input: "move forward 3 meters"
Output: robot.move_forward(distance=3.0, speed=1.0)

Input: "turn left 90 degrees"  
Output: robot.turn(direction="left", angle=90, speed=0.5)

Input: "go to position 5, 2, 1"
Output: robot.move_to(x=5.0, y=2.0, z=1.0)
```

### Object Manipulation

```python
Input: "pick up the red ball"
Output: robot.gripper.pick_up(object_type="ball", color="red", size="medium")

Input: "grab that wrench carefully"
Output: robot.gripper.grab(object_name="wrench", force=0.3)

Input: "drop it on the table"
Output: robot.gripper.drop(location="table")
```

### Sensor Operations

```python
Input: "scan the room for obstacles"
Output: robot.sensors.scan(range=10.0, angle=360)

Input: "detect any bottles nearby"
Output: robot.sensors.detect(object_type="bottles")

Input: "check distance to the wall ahead"
Output: robot.sensors.measure_distance(direction="forward")
```

## 🔧 Dataset Generation Examples

### Custom Dataset Configuration

```python
# Generate a smaller dataset for testing
dataset_config = {
    'movement_count': 1000,
    'manipulation_count': 800,
    'sensors_count': 500,
    'navigation_count': 500,
    'complex_count': 200,
    'conversational_count': 200,
    'error_handling_count': 100,
    'chained_complex_count': 50,
    'contextual_count': 300,
    'parametric_count': 250,
    'safety_count': 200,
    'efficiency_count': 150,
}

# Generate with custom config
generator = AsyncOpenAIDatasetGenerator(API_KEY, config)
dataset = await generator.generate_full_dataset_async(**dataset_config)
```

### Rate-Limited Generation

```python
# Conservative settings for API rate limits
config = AsyncGenerationConfig(
    examples_per_batch=25,           # Smaller batches
    max_concurrent_requests=2,       # Very conservative
    delay_between_batches=2.0,       # Longer delays
    request_delay=1.0,               # Slower requests
    model="gpt-4o-mini"
)
```

## 🎯 Training Examples

### Basic Training

```bash
# Train with default settings
python train_robot.py --data-json robot_dataset.json --output-dir ./models/basic-robot

# Train with custom parameters
python train_robot.py \
    --data-json robot_dataset.json \
    --output-dir ./models/custom-robot \
    --num-epochs 5 \
    --batch-size 8 \
    --learning-rate 2e-4 \
    --lora-r 64
```

### Advanced Training Configuration

```python
# Custom model configuration
config = ModelCfg(
    model_name="codellama/CodeLlama-13b-Python-hf",  # Larger model
    max_length=512,                                   # Longer sequences
    lora_r=64,                                       # Higher rank
    lora_alpha=128,                                  # Higher alpha
    batch_size=2,                                    # Smaller batch for memory
    learning_rate=5e-5,                              # Lower learning rate
    num_epochs=5                                     # More epochs
)
```

### Training with Validation Split

```bash
# Use separate train/validation files
python train_robot.py \
    --train-json robot_train_dataset.json \
    --val-json robot_val_dataset.json \
    --output-dir ./models/validated-robot
```

## 🎮 Demo Examples

### Basic Demo Launch

```bash
# Launch with default settings
python robot_demo.py \
    --base-model codellama/CodeLlama-7b-Python-hf \
    --lora-path ./models/robot-lora

# Launch with custom host/port
python robot_demo.py \
    --base-model codellama/CodeLlama-7b-Python-hf \
    --lora-path ./models/robot-lora \
    --host 0.0.0.0 \
    --port 7860
```

### Interactive Demo Usage

Once the demo is running, try these example inputs:

```
🤖 Simple Commands:
- "move forward 2 meters"
- "turn right 45 degrees"
- "pick up the blue cup"
- "scan for obstacles"

🔗 Complex Commands:
- "go to the kitchen and pick up any red objects"
- "scan the area, then grab the largest item you detect"
- "move forward 3 meters, turn left, then stop"

⚠️ Safety Commands:
- "move forward slowly and carefully"
- "pick that up very gently"
- "check if the path is clear before moving"

⚡ Efficiency Commands:
- "get to the living room as fast as possible"
- "take the shortest route to the kitchen"
- "pick up all the red objects quickly"
```

## 📊 Multi-Step Command Examples

### Complex Sequences

```python
Input: "go to kitchen, find cups, pick up a blue one, bring to living room"
Output:
robot.navigate.go_to_room(room="kitchen")
robot.sensors.detect(object_type="cups")
robot.gripper.pick_up(object_type="cup", color="blue", size="medium")
robot.navigate.go_to_room(room="living room")
robot.gripper.drop(location="living room")

Input: "scan for obstacles, move forward carefully, then pick up anything red"
Output:
robot.sensors.scan(range=5.0, angle=180)
robot.move_forward(distance=1.0, speed=0.3)
robot.sensors.detect(object_type="objects")
robot.gripper.pick_up(object_type="objects", color="red", size="any")
```

### Contextual Commands

```python
Input: "pick up the object you just scanned"
Output: robot.gripper.pick_up(object_type="detected_object", color="any", size="any")

Input: "go back to where you were before"
Output: robot.navigate.return_home()

Input: "repeat that last action three times"
Output:
robot.move_forward(distance=2.0, speed=1.0)
robot.move_forward(distance=2.0, speed=1.0)  
robot.move_forward(distance=2.0, speed=1.0)
```

## ⚠️ Error Handling Examples

### Correction Commands

```python
Input: "oops that's wrong, stop"
Output: robot.stop()

Input: "cancel that command"
Output: robot.stop()

Input: "wait, go back"
Output: robot.move_backward(distance=1.0, speed=0.5)

Input: "emergency stop now"
Output: robot.stop()
```

## 🔧 Advanced Usage

### Custom Robot API Extensions

If you need to extend the robot API, modify the prompts in `dataset_generator.py`:

```python
# Add new API methods to the base prompt
ROBOT_API_SPECIFICATION = """
- robot.move_forward(distance=float, speed=float)
- robot.move_backward(distance=float, speed=float)
- robot.turn(direction="left"|"right", angle=int, speed=float)
# ... existing methods ...

# NEW METHODS:
- robot.arm.extend(length=float)
- robot.arm.retract(length=float)
- robot.lights.turn_on(brightness=float)
- robot.lights.turn_off()
"""
```

### Integration with Real Robots

To integrate with actual robot hardware:

1. **Replace API calls** with real robot control functions
2. **Add safety validation** before executing commands
3. **Implement error handling** for hardware failures
4. **Add sensor feedback** for closed-loop control

```python
# Example integration wrapper
class RealRobotExecutor:
    def __init__(self, robot_interface):
        self.robot = robot_interface
    
    def execute_command(self, generated_code: str):
        # Parse and validate the generated code
        if self.validate_safety(generated_code):
            # Execute on real robot
            exec(generated_code)
        else:
            print("Safety check failed!")
```

## 📈 Performance Optimization

### Memory Optimization

```python
# For large datasets, use gradient checkpointing
training_args = TrainingArguments(
    gradient_checkpointing=True,
    dataloader_pin_memory=False,
    dataloader_num_workers=4,
    # ... other args
)
```

### Speed Optimization

```python
# Use compilation for faster training (PyTorch 2.0+)
model = torch.compile(model)

# Use SDPA attention for better memory efficiency
model.config.use_sdpa = True
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train_robot.py --batch-size 2
   ```

2. **Rate Limit Errors**
   ```python
   # Increase delays in config
   config.request_delay = 2.0
   config.max_concurrent_requests = 1
   ```

3. **Poor Generation Quality**
   ```bash
   # Train for more epochs with lower learning rate
   python train_robot.py --num-epochs 10 --learning-rate 1e-5
   ```

For more examples and detailed usage, see the main [README.md](README.md).
