import json
import re
from typing import List, Dict, Any

def fix_robot_command(command: str) -> str:
    """Attempt to fix common issues with robot commands"""
    command = command.strip()
    
    # If starts with robot. and has one '(' but missing ')', add it
    if (command.startswith('robot.') and 
        command.count('(') == 1 and 
        command.count(')') == 0):
        command = command + ')'
        print(f"  🔧 Fixed missing ')': {command}")
    
    return command

def is_valid_robot_command(command: str) -> bool:
    """Check if command is well-formed: 
    - starts with 'robot.'
    - ends with ')'
    - has exactly one '(' and one ')'
    - '(' comes before ')'
    """
    command = command.strip()
    
    # Basic format check
    if not (command.startswith('robot.') and command.endswith(')')):
        return False
    
    # Count parentheses
    open_count = command.count('(')
    close_count = command.count(')')
    
    if open_count != 1 or close_count != 1:
        return False
    
    # Check that '(' comes before ')'
    open_pos = command.find('(')
    close_pos = command.rfind(')')
    
    if open_pos >= close_pos:
        return False
    
    return True

def split_multi_commands(command: str) -> list:
    """Split a command string that contains multiple robot commands separated by \n or ;"""
    import re
    
    # First, convert literal \n strings to actual newlines
    command = command.replace('\\n', '\n')
    
    # Split on both newlines and semicolons
    parts = re.split(r'[\n;]', command)
    
    # Clean up each part and filter out empty strings
    split_commands = []
    for part in parts:
        part = part.strip()
        if part:  # Only add non-empty parts
            split_commands.append(part)
    
    if len(split_commands) > 1:
        print(f"    🔀 Split: '{command[:50]}...' → {len(split_commands)} commands")
    
    return split_commands

def preprocess_output_list(output_list: list) -> list:
    """Preprocess output_list to split multi-command elements"""
    expanded_list = []
    
    for item in output_list:
        if isinstance(item, str):
            # Split this item if it contains multiple commands
            split_items = split_multi_commands(item)
            expanded_list.extend(split_items)
        else:
            # Keep non-string items as-is (they'll be caught in validation)
            expanded_list.append(item)
    
    return expanded_list

def clean_robot_dataset(input_file: str, output_file: str) -> None:
    """Clean dataset and add EOS tokens"""
    
    # Use standard EOS token that CodeLlama already knows
    EOS_TOKEN = "<|endoftext|>"
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    cleaned_data = []
    stats = {
        'total_examples': len(data),
        'valid_examples': 0,
        'removed_examples': 0,
        'malformed_commands': 0,
        'fixed_commands': 0,
        'split_commands': 0
    }
    
    for i, example in enumerate(data):
        try:
            output_list = example.get('output_list', [])
            
            # STEP 1: Preprocess - split multi-command elements
            original_length = len(output_list)
            expanded_output_list = preprocess_output_list(output_list)
            
            if len(expanded_output_list) > original_length:
                split_count = len(expanded_output_list) - original_length
                stats['split_commands'] += split_count
                print(f"  📦 Example {i}: Split {original_length} → {len(expanded_output_list)} commands")
            
            # STEP 2: Check and fix each command in expanded output_list
            valid_commands = []
            has_unfixable = False
            
            for cmd in expanded_output_list:
                if isinstance(cmd, str):
                    # Try to fix the command first
                    original_cmd = cmd
                    fixed_cmd = fix_robot_command(cmd)
                    
                    if fixed_cmd != original_cmd:
                        stats['fixed_commands'] += 1
                    
                    if is_valid_robot_command(fixed_cmd):
                        valid_commands.append(fixed_cmd)
                    else:
                        print(f"⚠️  Example {i}: Unfixable command: '{cmd}'")
                        stats['malformed_commands'] += 1
                        has_unfixable = True
                else:
                    print(f"⚠️  Example {i}: Non-string command: {cmd}")
                    stats['malformed_commands'] += 1
                    has_unfixable = True
            
            # Only keep examples where ALL commands are valid
            if valid_commands and not has_unfixable:
                # Add EOS token at the end
                valid_commands.append(EOS_TOKEN)
                
                # Update the example
                cleaned_example = example.copy()
                cleaned_example['output_list'] = valid_commands
                
                # Also update the 'output' field to include EOS
                cleaned_example['output'] = '\n'.join(valid_commands[:-1]) + '\n' + EOS_TOKEN
                
                cleaned_data.append(cleaned_example)
                stats['valid_examples'] += 1
            else:
                print(f"❌ Removing example {i}: '{example['input'][:50]}...'")
                stats['removed_examples'] += 1
                
        except Exception as e:
            print(f"❌ Error processing example {i}: {e}")
            stats['removed_examples'] += 1
    
    # Save cleaned dataset
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    # Print statistics
    print(f"\n📊 Cleaning Results:")
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Valid examples: {stats['valid_examples']}")
    print(f"   Removed examples: {stats['removed_examples']}")
    print(f"   Commands split: {stats['split_commands']}")
    print(f"   Fixed commands: {stats['fixed_commands']}")
    print(f"   Malformed commands found: {stats['malformed_commands']}")
    print(f"   Success rate: {stats['valid_examples']/stats['total_examples']*100:.1f}%")
    print(f"   EOS token used: '{EOS_TOKEN}'")
    print(f"\n✅ Cleaned dataset saved to: {output_file}")

def validate_dataset(file_path: str) -> None:
    """Validate the cleaned dataset"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"\n🔍 Validating {len(data)} examples...")
    
    for i, example in enumerate(data[:5]):  # Check first 5
        print(f"\nExample {i}:")
        print(f"  Input: {example['input'][:50]}...")
        print(f"  Commands: {len(example['output_list'])-1} + EOS")  # -1 for EOS token
        for cmd in example['output_list']:
            if cmd == "<|endoftext|>":
                print(f"    {cmd} ← EOS token")
            else:
                print(f"    {cmd}")

if __name__ == "__main__":
    input_file = "robot_full_dataset_list.json"
    output_file = "robot_clean_dataset_list.json"
    
    print("🧹 Cleaning robot dataset...")
    clean_robot_dataset(input_file, output_file)
    
    print("\n🔍 Validating cleaned dataset...")
    validate_dataset(output_file)