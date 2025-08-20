#!/usr/bin/env python3
"""
Quick Start Example for Robot-LoRA

This script demonstrates basic usage of the Robot-LoRA system:
1. Generate a small dataset
2. Train a model
3. Test the trained model

Usage:
    python examples/quick_start.py
"""

import os
import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_generator import AsyncOpenAIDatasetGenerator, AsyncGenerationConfig


async def generate_small_dataset():
    """Generate a small dataset for testing"""
    print("🔍 Checking for OpenAI API key...")
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your_key_here'")
        return None
    
    print("📊 Generating small test dataset...")
    
    # Small dataset configuration for quick testing
    dataset_config = {
        'movement_count': 50,
        'manipulation_count': 40,
        'sensors_count': 30,
        'navigation_count': 30,
        'complex_count': 20,
        'conversational_count': 20,
        'error_handling_count': 10,
        'chained_complex_count': 10,
        'contextual_count': 20,
        'parametric_count': 15,
        'safety_count': 15,
        'efficiency_count': 15,
    }
    
    # Conservative configuration for testing
    config = AsyncGenerationConfig(
        examples_per_batch=10,           # Small batches
        max_concurrent_requests=2,       # Conservative
        delay_between_batches=1.0,
        request_delay=0.5,
        model="gpt-4o-mini"
    )
    
    generator = AsyncOpenAIDatasetGenerator(api_key, config)
    
    print(f"🚀 Generating {sum(dataset_config.values())} examples...")
    dataset = await generator.generate_full_dataset_async(**dataset_config)
    
    # Save the dataset
    generator.save_dataset(dataset, 'quick_start_dataset.json')
    
    # Create train/val split
    train_data, val_data = generator.create_train_val_split(dataset, val_ratio=0.2)
    generator.save_dataset(train_data, 'quick_start_train.json')
    generator.save_dataset(val_data, 'quick_start_val.json')
    
    print(f"✅ Dataset generation complete!")
    print(f"📁 Files created:")
    print(f"   - quick_start_dataset.json ({len(dataset)} examples)")
    print(f"   - quick_start_train.json ({len(train_data)} examples)")
    print(f"   - quick_start_val.json ({len(val_data)} examples)")
    
    return dataset


def train_quick_model():
    """Train a small model for testing"""
    print("\n🎯 Starting model training...")
    print("💡 This will train a small LoRA model for testing")
    
    # Check if dataset exists
    if not os.path.exists('quick_start_train.json'):
        print("❌ No training dataset found. Please run dataset generation first.")
        return False
    
    # Import training modules
    try:
        from train_robot import main as train_main
        import sys
        
        # Set up training arguments
        sys.argv = [
            'train_robot.py',
            '--train-json', 'quick_start_train.json',
            '--val-json', 'quick_start_val.json',
            '--output-dir', './models/quick-start-robot',
            '--num-epochs', '2',
            '--batch-size', '2',
            '--learning-rate', '1e-4',
            '--save-steps', '50',
            '--eval-steps', '50'
        ]
        
        # Run training
        train_main()
        print("✅ Training complete!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 You can manually train with:")
        print("   python train_robot.py --train-json quick_start_train.json --val-json quick_start_val.json --output-dir ./models/quick-start-robot")
        return False


def test_model():
    """Test the trained model"""
    print("\n🧪 Testing the trained model...")
    
    model_path = "./models/quick-start-robot"
    if not os.path.exists(model_path):
        print("❌ No trained model found. Please train a model first.")
        print("💡 You can manually test with:")
        print("   python robot_demo.py --base-model codellama/CodeLlama-7b-Python-hf --lora-path ./models/quick-start-robot")
        return
    
    print("🎮 To test your model, run:")
    print(f"   python robot_demo.py --base-model codellama/CodeLlama-7b-Python-hf --lora-path {model_path}")
    print("\n🤖 Try these example commands in the demo:")
    print('   - "move forward 3 meters"')
    print('   - "pick up the red ball"')
    print('   - "go to the kitchen and scan for cups"')


async def main():
    """Main quick start workflow"""
    print("🚀 Robot-LoRA Quick Start")
    print("=" * 50)
    
    # Step 1: Generate dataset
    dataset = await generate_small_dataset()
    if not dataset:
        return
    
    # Step 2: Train model
    print("\n" + "=" * 50)
    train_success = train_quick_model()
    
    # Step 3: Test instructions
    print("\n" + "=" * 50)
    test_model()
    
    print("\n🎉 Quick start complete!")
    print("📖 See README.md for more detailed usage instructions")


if __name__ == "__main__":
    asyncio.run(main())
