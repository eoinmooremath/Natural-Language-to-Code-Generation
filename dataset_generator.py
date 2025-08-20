import asyncio
import aiohttp
import json
import time
import random
from typing import List, Dict, Optional
import re
from dataclasses import dataclass
import os

@dataclass
class AsyncGenerationConfig:
    """Configuration for async dataset generation"""
    examples_per_batch: int = 50      # Examples per API call
    max_concurrent_requests: int = 3   # Stay well under rate limits
    max_retries: int = 3
    delay_between_batches: float = 1.0  # 1 second delay between batches
    request_delay: float = 0.5         # 0.5 second delay between individual requests
    model: str = "gpt-5-mini"          # Back to gpt-5-mini
    timeout: int = 60                  # Longer timeout for rate-limited requests

class AsyncOpenAIDatasetGenerator:
    def __init__(self, api_key: str, config: AsyncGenerationConfig = None):
        """Initialize with OpenAI API key"""
        self.api_key = api_key
        self.config = config or AsyncGenerationConfig()
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
    def create_base_prompt(self) -> str:
        """Create the base system prompt for consistent generation"""
        return """You are an expert at creating training data for robot control systems. 

Generate natural language commands that humans would give to robots, paired with corresponding Python robot API calls.

ROBOT API SPECIFICATION:
- robot.move_forward(distance=float, speed=float)
- robot.move_backward(distance=float, speed=float) 
- robot.turn(direction="left"|"right"|"clockwise"|"counterclockwise", angle=int, speed=float)
- robot.move_to(x=float, y=float, z=float)
- robot.stop()
- robot.gripper.pick_up(object_type=str, color=str, size=str)
- robot.gripper.drop(location=str)
- robot.gripper.grab(object_name=str, force=float)
- robot.gripper.release()
- robot.sensors.scan(range=float, angle=int)
- robot.sensors.detect(object_type=str)
- robot.sensors.measure_distance(direction=str)
- robot.navigate.go_to_room(room=str)
- robot.navigate.follow_path(path_name=str, speed=float)
- robot.navigate.return_home()

IMPORTANT GUIDELINES:
1. Generate DIVERSE natural language - formal, casual, polite, urgent, conversational
2. Include realistic parameters (distances 0.1-10m, speeds 0.1-2.0, angles 30-360°)
3. Use real object types, colors, room names
4. Include typos and informal language occasionally
5. Some commands can be multi-line for complex actions
6. Return valid JSON format only

Example output format:
[
  {"input": "move forward 3 meters", "output": "robot.move_forward(distance=3.0, speed=1.0)"},
  {"input": "can you please turn left?", "output": "robot.turn(direction=\"left\", angle=90, speed=0.5)"}
]"""

    def create_category_prompts(self) -> Dict[str, str]:
        """Create specialized prompts for different command categories"""
        base_prompt = self.create_base_prompt()
        
        return {
            "movement": f"""{base_prompt}

FOCUS: Generate movement commands (forward, backward, turning, positioning, stopping)
Include variations like:
- "go forward 2 steps" 
- "back up slowly"
- "turn around"
- "move to position 5, 3"
- "stop right now"
- "spin clockwise 180 degrees"

Generate {self.config.examples_per_batch} diverse movement examples:""",

            "manipulation": f"""{base_prompt}

FOCUS: Generate object manipulation commands (picking, dropping, grabbing)
Include variations like:
- "pick up the red ball"
- "grab that wrench carefully"  
- "drop it on the table"
- "release the object"
- "lift the small blue box"

Generate {self.config.examples_per_batch} diverse manipulation examples:""",

            "sensors": f"""{base_prompt}

FOCUS: Generate sensor and detection commands
Include variations like:
- "scan the room"
- "look for obstacles" 
- "check distance ahead"
- "detect any bottles"
- "measure how far the wall is"

Generate {self.config.examples_per_batch} diverse sensor examples:""",

            "navigation": f"""{base_prompt}

FOCUS: Generate navigation and room movement commands  
Include variations like:
- "go to the kitchen"
- "navigate to bedroom"
- "follow path A"
- "return to base"
- "head home"

Generate {self.config.examples_per_batch} diverse navigation examples:""",

            "complex": f"""{base_prompt}

FOCUS: Generate multi-step commands that require 2-3 robot actions
Include variations like:
- "pick up the cup and bring it to the kitchen"
- "scan for objects then grab the red one"
- "go to the living room and pick up any books"
- "turn left, move forward 2 meters, then stop"

For multi-step commands, separate each action on a new line in the output.

Generate {self.config.examples_per_batch} diverse complex examples:""",

            "error_handling": f"""{base_prompt}

FOCUS: Generate error handling and correction commands
Include variations like:
- "oops that's wrong, stop"
- "cancel that command"
- "undo the last action"
- "wait, go back"
- "that's not right, try again"
- "abort mission"
- "emergency stop"

Generate {self.config.examples_per_batch} diverse error handling examples:""",

            "contextual": f"""{base_prompt}

FOCUS: Generate context-aware commands that reference environment or previous actions
Include variations like:
- "pick up the object you just scanned"
- "go back to where you were before"
- "grab the same color cup as last time"
- "move to the brightest area you detected"
- "follow the same path but in reverse"
- "repeat that last action three times"

Generate {self.config.examples_per_batch} diverse contextual examples:""",

            "parametric": f"""{base_prompt}

FOCUS: Generate commands with many specific parameters and measurements
Include variations like:
- "move forward exactly 2.7 meters at 0.8 m/s"
- "turn 127 degrees clockwise at half speed"
- "pick up object with 0.3 newton force"
- "scan 270 degrees with 15 meter range"
- "move to coordinates -3.2, 4.7, 1.1"

Generate {self.config.examples_per_batch} diverse parametric examples:""",

            "safety": f"""{base_prompt}

FOCUS: Generate safety-conscious and cautious commands
Include variations like:
- "move forward slowly and carefully"
- "pick that up very gently"
- "scan for obstacles before moving"
- "check if the path is clear"
- "move at minimum safe speed"
- "be extra careful with that fragile item"

Generate {self.config.examples_per_batch} diverse safety examples:""",

            "chained_complex": f"""{base_prompt}

FOCUS: Generate long multi-step command sequences (4-6 actions)
Include variations like:
- "go to kitchen, scan for cups, pick up a blue one, bring it to living room, place on table"
- "turn left, move forward 3 meters, scan area, detect objects, pick up any red items"
- "navigate to bedroom, look for books, grab the largest one, return to office, drop on desk"

For chained commands, separate each action on a new line in the output.

Generate {self.config.examples_per_batch} diverse chained complex examples:""",

            "contextual": f"""{base_prompt}

FOCUS: Generate context-aware commands that reference environment or previous actions
Include variations like:
- "pick up the object you just scanned"
- "go back to where you were before"
- "grab the same color cup as last time"
- "move to the brightest area you detected"
- "follow the same path but in reverse"
- "repeat that last action three times"

Generate {self.config.examples_per_batch} diverse contextual examples:""",

            "parametric": f"""{base_prompt}

FOCUS: Generate commands with many specific parameters and measurements
Include variations like:
- "move forward exactly 2.7 meters at 0.8 m/s"
- "turn 127 degrees clockwise at half speed"
- "pick up object with 0.3 newton force"
- "scan 270 degrees with 15 meter range"
- "move to coordinates -3.2, 4.7, 1.1"

Generate {self.config.examples_per_batch} diverse parametric examples:""",

            "safety": f"""{base_prompt}

FOCUS: Generate safety-conscious and cautious commands
Include variations like:
- "move forward slowly and carefully"
- "pick that up very gently"
- "scan for obstacles before moving"
- "check if the path is clear"
- "move at minimum safe speed"
- "be extra careful with that fragile item"

Generate {self.config.examples_per_batch} diverse safety examples:""",

            "efficiency": f"""{base_prompt}

FOCUS: Generate speed and efficiency focused commands
Include variations like:
- "get there as fast as possible"
- "pick up all the red objects quickly"
- "take the shortest route to kitchen"
- "maximize speed while staying safe"
- "do that as efficiently as you can"
- "optimize the path and move fast"

Generate {self.config.examples_per_batch} diverse efficiency examples:"""
        }

    def parse_llm_response(self, response_text: str) -> List[Dict]:
        """Parse LLM response and extract valid examples"""
        try:
            # Try to parse as JSON directly
            if response_text.strip().startswith('['):
                return json.loads(response_text)
            
            # Look for JSON array in the response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Try to extract individual JSON objects
            examples = []
            json_objects = re.findall(r'\{[^}]+\}', response_text)
            for obj_str in json_objects:
                try:
                    example = json.loads(obj_str)
                    if 'input' in example and 'output' in example:
                        examples.append(example)
                except:
                    continue
            
            return examples
            
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return []

    def validate_example(self, example: Dict) -> bool:
        """Validate that an example is properly formatted"""
        if not isinstance(example, dict):
            return False
        
        if 'input' not in example or 'output' not in example:
            return False
        
        if not example['input'].strip() or not example['output'].strip():
            return False
        
        # Basic validation that output looks like robot code
        output = example['output']
        if not ('robot.' in output and '(' in output):
            return False
        
        return True

    async def make_api_request(self, session: aiohttp.ClientSession, prompt: str, category: str, batch_id: int) -> List[Dict]:
        """Make a single async API request with rate limiting"""
        # Add delay before each request to respect rate limits
        await asyncio.sleep(self.config.request_delay)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that generates training data in JSON format."},
                {"role": "user", "content": prompt}
            ]
            # "max_tokens": 2000
        }
        
        for attempt in range(self.config.max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self.config.timeout)
                async with session.post(self.base_url, json=payload, headers=headers, timeout=timeout) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result['choices'][0]['message']['content']
                        examples = self.parse_llm_response(response_text)
                        
                        # Validate examples
                        valid_examples = []
                        for example in examples:
                            if self.validate_example(example):
                                example['category'] = category
                                valid_examples.append(example)
                        
                        print(f"✓ {category} batch {batch_id}: {len(valid_examples)} examples")
                        return valid_examples
                    
                    elif response.status == 429:  # Rate limit
                        wait_time = min(60, 5 * (2 ** attempt))  # Cap at 60 seconds
                        print(f"⚠️  Rate limit hit for {category} batch {batch_id}, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"❌ API error {response.status}: {await response.text()}")
                        
            except asyncio.TimeoutError:
                print(f"⚠️  Timeout on {category} batch {batch_id}, attempt {attempt + 1}")
            except Exception as e:
                print(f"❌ Error on {category} batch {batch_id}: {e}")
                
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        print(f"❌ Failed to generate {category} batch {batch_id} after {self.config.max_retries} attempts")
        return []

    async def generate_category_dataset_async(self, category: str, total_examples: int) -> List[Dict]:
        """Generate dataset for a specific category using async requests"""
        prompts = self.create_category_prompts()
        base_prompt = prompts[category]
        
        batches_needed = (total_examples + self.config.examples_per_batch - 1) // self.config.examples_per_batch
        
        print(f"\n🚀 Generating {total_examples} {category} examples ({batches_needed} batches)...")
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def bounded_request(session, prompt, batch_id):
            async with semaphore:
                return await self.make_api_request(session, prompt, category, batch_id)
        
        # Create all request tasks
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            
            for batch_id in range(batches_needed):
                # Add variety to prompts for later batches
                varied_prompt = base_prompt
                if batch_id > 0:
                    varied_prompt += f"\n\nNote: Make these examples different from previous batches. Be creative with phrasing and scenarios. Batch {batch_id + 1}."
                
                task = bounded_request(session, varied_prompt, batch_id + 1)
                tasks.append(task)
                
                # Small delay between task creation to avoid overwhelming
                if batch_id > 0 and batch_id % 5 == 0:
                    await asyncio.sleep(self.config.delay_between_batches)
            
            # Execute all requests concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all valid examples
        all_examples = []
        for result in results:
            if isinstance(result, list):
                all_examples.extend(result)
            elif isinstance(result, Exception):
                print(f"❌ Task failed with exception: {result}")
        
        # Trim to exact number requested
        final_examples = all_examples[:total_examples]
        print(f"✅ {category} complete: {len(final_examples)} examples")
        return final_examples

    async def generate_full_dataset_async(self, 
                                        movement_count: int = 15000,
                                        manipulation_count: int = 12000,
                                        sensors_count: int = 8000,
                                        navigation_count: int = 8000,
                                        complex_count: int = 6000,
                                        conversational_count: int = 6000,
                                        error_handling_count: int = 3000,
                                        chained_complex_count: int = 2000,
                                        contextual_count: int = 5000,
                                        parametric_count: int = 4000,
                                        safety_count: int = 3000,
                                        efficiency_count: int = 3000) -> List[Dict]:
        """Generate complete dataset across all categories using async"""
        
        categories = {
            'movement': movement_count,
            'manipulation': manipulation_count, 
            'sensors': sensors_count,
            'navigation': navigation_count,
            'complex': complex_count,
            'conversational': conversational_count,
            'error_handling': error_handling_count,
            'chained_complex': chained_complex_count,
            'contextual': contextual_count,
            'parametric': parametric_count,
            'safety': safety_count,
            'efficiency': efficiency_count
        }
        
        total_requested = sum(categories.values())
        print(f"🎯 Starting ASYNC dataset generation for {total_requested} total examples...")
        print(f"⚡ Max concurrent requests: {self.config.max_concurrent_requests}")
        start_time = time.time()
        
        # Generate all categories concurrently
        tasks = []
        for category, count in categories.items():
            if count > 0:
                task = self.generate_category_dataset_async(category, count)
                tasks.append((category, task))
        
        # Execute all category generation concurrently
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        # Combine results
        all_examples = []
        for i, result in enumerate(results):
            category = tasks[i][0]
            if isinstance(result, list):
                all_examples.extend(result)
                print(f"✅ {category}: {len(result)} examples added")
            elif isinstance(result, Exception):
                print(f"❌ {category} failed: {result}")
        
        # Shuffle the final dataset
        random.shuffle(all_examples)
        
        elapsed_time = time.time() - start_time
        print(f"\n🎉 ASYNC dataset generation complete!")
        print(f"📊 Generated {len(all_examples)} total examples in {elapsed_time:.1f} seconds")
        print(f"⚡ Speed: {len(all_examples)/elapsed_time:.1f} examples/second")
        print(f"🚀 {elapsed_time/60:.1f}x faster than sequential!")
        
        return all_examples

    def save_dataset(self, dataset: List[Dict], filepath: str):
        """Save dataset to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"💾 Dataset saved to {filepath}")

    def create_train_val_split(self, dataset: List[Dict], val_ratio: float = 0.2):
        """Create train/validation split"""
        random.shuffle(dataset)
        split_idx = int(len(dataset) * (1 - val_ratio))
        
        train_data = dataset[:split_idx]
        val_data = dataset[split_idx:]
        
        return train_data, val_data

    def print_sample_examples(self, dataset: List[Dict], num_examples: int = 8):
        """Print sample examples from the dataset"""
        print(f"\n{'='*60}")
        print(f"SAMPLE EXAMPLES ({num_examples} random examples)")
        print(f"{'='*60}")
        
        sample_data = random.sample(dataset, min(num_examples, len(dataset)))
        
        for i, example in enumerate(sample_data, 1):
            print(f"\n[{i}] Category: {example.get('category', 'unknown')}")
            print(f"Input:  \"{example['input']}\"")
            print(f"Output: {example['output']}")
            print("-" * 50)

    def analyze_dataset(self, dataset: List[Dict]):
        """Analyze dataset statistics"""
        print(f"\n{'='*60}")
        print("DATASET ANALYSIS")
        print(f"{'='*60}")
        
        # Category breakdown
        category_counts = {}
        total_chars_input = 0
        total_chars_output = 0
        
        for example in dataset:
            category = example.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
            total_chars_input += len(example['input'])
            total_chars_output += len(example['output'])
        
        print(f"Total examples: {len(dataset)}")
        print(f"Average input length: {total_chars_input / len(dataset):.1f} characters")
        print(f"Average output length: {total_chars_output / len(dataset):.1f} characters")
        
        print(f"\nCategory breakdown:")
        for category, count in sorted(category_counts.items()):
            percentage = (count / len(dataset)) * 100
            print(f"  {category:>15}: {count:>4} examples ({percentage:>5.1f}%)")

def estimate_cost_and_time(total_examples: int, examples_per_batch: int = 20, concurrent_requests: int = 10) -> tuple:
    """Estimate OpenAI API cost and time for async generation"""
    # Cost estimation (same as before)
    prompt_tokens_per_batch = 800
    completion_tokens_per_batch = 400
    batches = (total_examples + examples_per_batch - 1) // examples_per_batch
    
    total_prompt_tokens = batches * prompt_tokens_per_batch
    total_completion_tokens = batches * completion_tokens_per_batch
    
    cost_per_1k_prompt = 0.00015
    cost_per_1k_completion = 0.0006
    
    estimated_cost = (total_prompt_tokens / 1000 * cost_per_1k_prompt + 
                     total_completion_tokens / 1000 * cost_per_1k_completion)
    
    # Time estimation with async
    avg_request_time = 3.0  # seconds per request
    sequential_time = batches * avg_request_time
    async_time = (batches / concurrent_requests) * avg_request_time
    
    return estimated_cost, sequential_time, async_time

async def main():
    """Main async function"""
    # Configuration
    API_KEY = os.getenv('OPENAI_API_KEY')
    if not API_KEY:
        print("❌ Please set OPENAI_API_KEY environment variable")
        return
    
    # Dataset size configuration - PRODUCTION SCALE for serious training
    dataset_config = {
        'movement_count': 15000,        # Core movement commands
        'manipulation_count': 12000,    # Object manipulation  
        'sensors_count': 8000,          # Sensor operations
        'navigation_count': 8000,       # Room/path navigation
        'complex_count': 6000,          # Multi-step sequences
        'conversational_count': 6000,   # Natural conversation
        'error_handling_count': 3000,   # Error correction
        'chained_complex_count': 2000,  # Long sequences (4-6 steps)
        'contextual_count': 5000,       # NEW: Context-aware commands
        'parametric_count': 4000,       # NEW: Parameter-heavy commands
        'safety_count': 3000,           # NEW: Safety and caution commands
        'efficiency_count': 3000,       # NEW: Speed/efficiency focused
    }
    # TOTAL: ~75,000 examples
    
    total_examples = sum(dataset_config.values())
    estimated_cost, sequential_time, async_time = estimate_cost_and_time(total_examples, concurrent_requests=15)
    
    print(f"🎯 PRODUCTION-SCALE Dataset Generation Plan:")
    print(f"📊 Total examples: {total_examples:,}")
    print(f"💰 Estimated cost: ${estimated_cost:.2f}")
    print(f"⏱️  Sequential time: {sequential_time/3600:.1f} hours")
    print(f"⚡ Async time: {async_time/3600:.1f} hours ({sequential_time/async_time:.1f}x faster!)")
    print(f"🔥 This will create a POWERFUL training dataset!")
    
    confirm = input(f"\n🚀 Proceed with generating {total_examples:,} examples? (y/n): ")
    if confirm.lower() != 'y':
        print("❌ Generation cancelled.")
        return
    
    # Create async generator with rate-limited settings to avoid API limits
    config = AsyncGenerationConfig(
        examples_per_batch=20,           # Reasonable batch size
        max_concurrent_requests=3,       # Conservative to avoid rate limits 
        delay_between_batches=1.0,       # 1 second between batches
        request_delay=0.5                # 0.5 second between requests
    )
    
    generator = AsyncOpenAIDatasetGenerator(API_KEY, config)
    
    # Generate dataset asynchronously
    dataset = await generator.generate_full_dataset_async(**dataset_config)
    
    # Analyze results
    generator.analyze_dataset(dataset)
    generator.print_sample_examples(dataset, num_examples=10)
    
    # Create train/val split
    train_data, val_data = generator.create_train_val_split(dataset, val_ratio=0.2)
    
    print(f"\n📊 Train/Validation Split:")
    print(f"🏋️  Training examples: {len(train_data)}")
    print(f"✅ Validation examples: {len(val_data)}")
    
    # Save datasets
    generator.save_dataset(train_data, 'robot_train_dataset.json')
    generator.save_dataset(val_data, 'robot_val_dataset.json') 
    generator.save_dataset(dataset, 'robot_full_dataset.json')
    
    print(f"\n🎉 ASYNC dataset generation complete!")
    print(f"📁 Files saved:")
    print(f"  - robot_full_dataset.json ({len(dataset)} examples)")
    print(f"  - robot_train_dataset.json ({len(train_data)} examples)")
    print(f"  - robot_val_dataset.json ({len(val_data)} examples)")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())