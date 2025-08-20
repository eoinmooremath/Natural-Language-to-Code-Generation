#!/usr/bin/env python3
"""
Custom Dataset Generation Example

This script shows how to create custom datasets with specific configurations
for different use cases and robot types.

Usage:
    python examples/generate_custom_dataset.py --config mobile_robot
    python examples/generate_custom_dataset.py --config industrial_arm
    python examples/generate_custom_dataset.py --config service_robot
"""

import os
import asyncio
import argparse
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_generator import AsyncOpenAIDatasetGenerator, AsyncGenerationConfig


# Predefined configurations for different robot types
ROBOT_CONFIGS = {
    "mobile_robot": {
        "description": "Mobile robot focused on navigation and basic manipulation",
        "dataset_config": {
            'movement_count': 8000,
            'manipulation_count': 3000,
            'sensors_count': 4000,
            'navigation_count': 6000,
            'complex_count': 2000,
            'conversational_count': 2000,
            'error_handling_count': 1000,
            'chained_complex_count': 500,
            'contextual_count': 1500,
            'parametric_count': 1000,
            'safety_count': 1500,
            'efficiency_count': 1500,
        }
    },
    
    "industrial_arm": {
        "description": "Industrial robotic arm focused on precision manipulation",
        "dataset_config": {
            'movement_count': 3000,
            'manipulation_count': 10000,
            'sensors_count': 3000,
            'navigation_count': 1000,
            'complex_count': 4000,
            'conversational_count': 2000,
            'error_handling_count': 2000,
            'chained_complex_count': 1500,
            'contextual_count': 2000,
            'parametric_count': 3000,
            'safety_count': 3000,
            'efficiency_count': 1500,
        }
    },
    
    "service_robot": {
        "description": "Service robot for household and office tasks",
        "dataset_config": {
            'movement_count': 6000,
            'manipulation_count': 6000,
            'sensors_count': 4000,
            'navigation_count': 6000,
            'complex_count': 4000,
            'conversational_count': 6000,
            'error_handling_count': 2000,
            'chained_complex_count': 2000,
            'contextual_count': 4000,
            'parametric_count': 2000,
            'safety_count': 2000,
            'efficiency_count': 2000,
        }
    },
    
    "research_robot": {
        "description": "Research robot with advanced capabilities",
        "dataset_config": {
            'movement_count': 5000,
            'manipulation_count': 5000,
            'sensors_count': 5000,
            'navigation_count': 5000,
            'complex_count': 5000,
            'conversational_count': 4000,
            'error_handling_count': 2000,
            'chained_complex_count': 2000,
            'contextual_count': 3000,
            'parametric_count': 3000,
            'safety_count': 2000,
            'efficiency_count': 2000,
        }
    },
    
    "micro": {
        "description": "Minimal dataset for testing (fast generation)",
        "dataset_config": {
            'movement_count': 100,
            'manipulation_count': 80,
            'sensors_count': 60,
            'navigation_count': 60,
            'complex_count': 40,
            'conversational_count': 40,
            'error_handling_count': 20,
            'chained_complex_count': 20,
            'contextual_count': 40,
            'parametric_count': 30,
            'safety_count': 30,
            'efficiency_count': 30,
        }
    }
}


async def generate_custom_dataset(config_name: str, output_prefix: str = None):
    """Generate a custom dataset based on predefined configuration"""
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return None
    
    # Get configuration
    if config_name not in ROBOT_CONFIGS:
        print(f"❌ Unknown configuration: {config_name}")
        print(f"Available configurations: {list(ROBOT_CONFIGS.keys())}")
        return None
    
    robot_config = ROBOT_CONFIGS[config_name]
    dataset_config = robot_config["dataset_config"]
    total_examples = sum(dataset_config.values())
    
    print(f"🤖 Generating dataset for: {robot_config['description']}")
    print(f"📊 Total examples: {total_examples:,}")
    
    # Set up file names
    if output_prefix is None:
        output_prefix = f"robot_{config_name}"
    
    # Configure generation settings based on dataset size
    if total_examples < 1000:
        # Fast settings for small datasets
        gen_config = AsyncGenerationConfig(
            examples_per_batch=20,
            max_concurrent_requests=3,
            delay_between_batches=0.5,
            request_delay=0.2,
            model="gpt-4o-mini"
        )
    elif total_examples < 10000:
        # Moderate settings for medium datasets
        gen_config = AsyncGenerationConfig(
            examples_per_batch=30,
            max_concurrent_requests=3,
            delay_between_batches=1.0,
            request_delay=0.3,
            model="gpt-4o-mini"
        )
    else:
        # Conservative settings for large datasets
        gen_config = AsyncGenerationConfig(
            examples_per_batch=40,
            max_concurrent_requests=3,
            delay_between_batches=1.0,
            request_delay=0.5,
            model="gpt-4o-mini"
        )
    
    # Create generator
    generator = AsyncOpenAIDatasetGenerator(api_key, gen_config)
    
    # Estimate time and cost
    estimated_cost, _, async_time = estimate_generation_time(total_examples, gen_config)
    print(f"💰 Estimated cost: ${estimated_cost:.2f}")
    print(f"⏱️  Estimated time: {async_time/3600:.1f} hours")
    
    # Confirm generation
    confirm = input(f"\n🚀 Proceed with generating {total_examples:,} examples? (y/n): ")
    if confirm.lower() != 'y':
        print("❌ Generation cancelled.")
        return None
    
    # Generate dataset
    print(f"\n🚀 Starting generation...")
    dataset = await generator.generate_full_dataset_async(**dataset_config)
    
    # Analyze and save
    generator.analyze_dataset(dataset)
    
    # Create files
    full_file = f"{output_prefix}_dataset.json"
    train_file = f"{output_prefix}_train.json"
    val_file = f"{output_prefix}_val.json"
    
    # Save full dataset
    generator.save_dataset(dataset, full_file)
    
    # Create train/val split
    train_data, val_data = generator.create_train_val_split(dataset, val_ratio=0.2)
    generator.save_dataset(train_data, train_file)
    generator.save_dataset(val_data, val_file)
    
    print(f"\n✅ Dataset generation complete!")
    print(f"📁 Files created:")
    print(f"   - {full_file} ({len(dataset)} examples)")
    print(f"   - {train_file} ({len(train_data)} examples)")
    print(f"   - {val_file} ({len(val_data)} examples)")
    
    # Show sample
    generator.print_sample_examples(dataset, num_examples=5)
    
    print(f"\n🎯 Next steps:")
    print(f"   python train_robot.py --train-json {train_file} --val-json {val_file} --output-dir ./models/{config_name}-robot")
    
    return dataset


def estimate_generation_time(total_examples: int, config: AsyncGenerationConfig):
    """Estimate generation time and cost"""
    # Cost estimation
    prompt_tokens_per_batch = 800
    completion_tokens_per_batch = 400
    batches = (total_examples + config.examples_per_batch - 1) // config.examples_per_batch
    
    total_prompt_tokens = batches * prompt_tokens_per_batch
    total_completion_tokens = batches * completion_tokens_per_batch
    
    # GPT-4o-mini pricing
    cost_per_1k_prompt = 0.00015
    cost_per_1k_completion = 0.0006
    
    estimated_cost = (total_prompt_tokens / 1000 * cost_per_1k_prompt + 
                     total_completion_tokens / 1000 * cost_per_1k_completion)
    
    # Time estimation
    avg_request_time = 3.0
    sequential_time = batches * avg_request_time
    async_time = (batches / config.max_concurrent_requests) * avg_request_time
    
    return estimated_cost, sequential_time, async_time


def list_configurations():
    """List all available configurations"""
    print("🤖 Available Robot Configurations:")
    print("=" * 50)
    
    for name, config in ROBOT_CONFIGS.items():
        total = sum(config["dataset_config"].values())
        print(f"\n{name}:")
        print(f"  Description: {config['description']}")
        print(f"  Total examples: {total:,}")
        
        # Show category breakdown
        categories = config["dataset_config"]
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top categories: {', '.join([f'{cat}({count})' for cat, count in top_categories])}")


def main():
    """Main function with CLI"""
    parser = argparse.ArgumentParser(description="Generate custom robot datasets")
    parser.add_argument("--config", type=str, 
                       help="Robot configuration to use",
                       choices=list(ROBOT_CONFIGS.keys()))
    parser.add_argument("--list", action="store_true",
                       help="List available configurations")
    parser.add_argument("--output-prefix", type=str,
                       help="Prefix for output files (default: robot_{config})")
    
    args = parser.parse_args()
    
    if args.list:
        list_configurations()
        return
    
    if not args.config:
        print("❌ Please specify a configuration with --config")
        print("💡 Use --list to see available configurations")
        return
    
    # Run generation
    asyncio.run(generate_custom_dataset(args.config, args.output_prefix))


if __name__ == "__main__":
    main()
