#!/usr/bin/env python3
import argparse
import time
from typing import Tuple

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel


# ---------- Stop rule to tame repetition ----------
class StopOnNonRobotOrMax(StoppingCriteria):
    """
    Stop generation when:
      - The last non-empty line doesn't start with 'robot.'
      - We reached max_lines lines
      - Or a blank line gap appears
    """
    def __init__(self, tokenizer, prompt_len: int, max_lines: int = 6):
        super().__init__()
        self.tok = tokenizer
        self.prompt_len = prompt_len
        self.max_lines = max_lines

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # Only check the first (single) sequence
        seq = input_ids[0].tolist()
        text = self.tok.decode(seq, skip_special_tokens=True)
        gen = text[self.prompt_len:].strip()

        # Split into non-empty lines
        lines = [ln.strip() for ln in gen.split("\n") if ln.strip()]
        if len(lines) >= self.max_lines:
            return True

        if lines:
            last = lines[-1]
            if not last.startswith("robot."):
                return True

        # Stop on double newline (blank paragraph)
        if gen.endswith("\n\n"):
            return True

        return False


# ---------- Demo ----------
class RobotComparisonDemo:
    def __init__(self, lora_model_path: str):
        self.lora_model_path = lora_model_path
        self.tokenizer = None
        self.base_model = None
        self.lora_model = None
        self._load_models()

    # --------- prompts ---------
    def create_base_prompt(self, command: str) -> str:
        return f"""You are a robot programming assistant. Convert natural language commands into robot API calls.

Rules:
- Output ONE robot.* action per line.
- Include ONLY the steps required by the command.
- If the command is a single atomic action, output exactly ONE line.
- No commentary or extra text.

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

Command: {command}

Robot code:
"""

    def create_lora_prompt(self, command: str) -> str:
        return (
            "### Instruction: Convert the command into robot API calls.\n"
            "- Output ONE action per line.\n"
            "- Include ONLY the steps required by the command.\n"
            "- If the command is a single atomic action, output exactly ONE line.\n"
            "- No commentary.\n\n"
            f"Command: {command}\n\n### Response:"
        )

    # --------- model loading ---------
    def _load_models(self):
        print("Loading models...")
        model_name = "codellama/CodeLlama-7b-Python-hf"

        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # base model
        print("📦 Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            trust_remote_code=True,
        ).eval()

        # peft adapter
        print("🎯 Loading LoRA fine-tuned adapter...")
        try:
            self.lora_model = PeftModel.from_pretrained(
                self.base_model,
                self.lora_model_path,
                torch_dtype=torch.bfloat16,
            ).eval()
            print("✅ LoRA loaded.")
        except Exception as e:
            print(f"❌ Error loading LoRA: {e}")
            print("Using base model for both comparisons.")
            self.lora_model = self.base_model

        # Diagnostics
        print("Active PEFT adapter:", getattr(self.lora_model, "active_adapter", None))
        has_lora = any(("lora_A" in n or "lora_B" in n) for n, _ in self.lora_model.named_parameters())
        print("LoRA layers present:", has_lora)
        print("Base model device:", next(self.base_model.parameters()).device)
        print("LoRA model device:", next(self.lora_model.parameters()).device)
        print("Are models the same object?", self.base_model is self.lora_model)

    # --------- helpers ---------
    def _clean_robot_block(self, text: str) -> str:
        """Return only the first contiguous block of lines that start with robot."""
        lines = []
        for ln in text.split("\n"):
            s = ln.strip()
            if not s:
                break
            if not s.startswith("robot."):
                break
            lines.append(s)
        return "\n".join(lines).strip()

    def _decode_response_body(self, full_text: str, prompt: str) -> str:
        """Prefer content after '### Response:'; else, strip prompt prefix."""
        if "### Response:" in full_text:
            body = full_text.split("### Response:", 1)[1].strip()
        else:
            body = full_text[len(prompt):].strip()
        return body

    # --------- generation paths ---------
    def generate_with_base_model(self, command: str) -> Tuple[str, float]:
        if not command.strip():
            return "Please enter a command", 0.0

        prompt = self.create_base_prompt(command)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # base: you can choose to sample lightly or keep deterministic; here we do light sampling
        start = time.time()
        with torch.no_grad():
            outputs = self.base_model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=160,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        dt = time.time() - start

        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        body = full[len(prompt):].strip()
        body = body.split("\n\n")[0].strip()  # chop trailing chatter
        return self._clean_robot_block(body), dt

    def generate_with_lora_model(self, command: str) -> Tuple[str, float]:
        if not command.strip():
            return "Please enter a command", 0.0

        prompt = self.create_lora_prompt(command)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        start = time.time()
        with torch.no_grad():
            outputs = self.lora_model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        dt = time.time() - start

        full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        body = self._decode_response_body(full, prompt)
        
        # Extract robot commands and stop at first repetition
        seen_lines = set()
        robot_lines = []
        
        for line in body.split("\n"):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Stop if we hit a non-robot line
            if not line.startswith("robot."):
                break
                
            # Stop if we've seen this exact line before
            if line in seen_lines:
                print(f"🔍 DEBUG: Stopping at repeated line: '{line}'")
                break
                
            # Add unique robot line
            seen_lines.add(line)
            robot_lines.append(line)
            
            # Safety limit - max 8 lines even if no repetition
            if len(robot_lines) >= 8:
                break
        
        result = "\n".join(robot_lines)
        return result, dt


    # --------- unified call for Gradio ---------
    def compare_models(self, command: str):
        # Run BASE first, then LoRA (sequential timing)
        # base_out, base_time = self.generate_with_base_model(command)
        # lora_out, lora_time = self.generate_with_lora_model(command)
        # return base_out, f"{base_time:.2f}s", lora_out, f"{lora_time:.2f}s"

        # Run BASE first, then LoRA (sequential timing)
        print(f"🔍 DEBUG: compare_models called with: '{command}'")
        
        # Run BASE first
        print("🔍 DEBUG: Running base model...")
        base_out, base_time = self.generate_with_base_model(command)
        print(f"🔍 DEBUG: Base result: '{base_out}', time: {base_time}")
        
        # Run LoRA second
        print("🔍 DEBUG: Running LoRA model...")
        lora_out, lora_time = self.generate_with_lora_model(command)
        print(f"🔍 DEBUG: LoRA result: '{lora_out}', time: {lora_time}")
        
        result = (base_out, f"{base_time:.2f}s", lora_out, f"{lora_time:.2f}s")
        print(f"🔍 DEBUG: Final return tuple: {result}")
        return result


def create_interface(lora_path: str):
    demo = RobotComparisonDemo(lora_path)

    samples = [
        "move forward 3 meters",
        "turn left 90 degrees",
        "pick up the red ball",
        "go to the kitchen",
        "scan for obstacles",
        "grab the blue cup gently",
        "move to coordinates 5, 3",
        "drop it on the table",
        "return home",
        "scan 6m, locate yellow ball, pick it up",
        "Go to the hallway, turn right 90 degrees, move forward 2 meters, stop."
    ]

    with gr.Blocks(title="Robot Model Comparison", theme=gr.themes.Soft()) as ui:
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 24px;">
          <h2>🤖 Robot Programming — Base vs LoRA</h2>
          <p>CodeLlama-7B-Python (base) vs your LoRA adapter</p>
        </div>
        """)

        with gr.Row():
            with gr.Column():
                cmd = gr.Textbox(
                    label="Natural Language Command",
                    value=samples[1],
                    placeholder="e.g., 'turn left 90 degrees'",
                    lines=2,
                )
                with gr.Row():
                    run = gr.Button("🚀 Compare", variant="primary")
                    use_sample = gr.Dropdown(choices=samples, value=samples[1], label="Samples")
                    set_sample = gr.Button("Use Sample")

        gr.Markdown("## 📊 Results")
        with gr.Group():
            gr.Markdown("### 🔴 Base Model (prompted)")
            with gr.Row():
                base_out = gr.Textbox(label="Generated Robot Code", lines=8, interactive=False)
                base_time = gr.Textbox(label="Time", lines=1, interactive=False)

        with gr.Group():
            gr.Markdown("### 🟢 LoRA Fine-tuned")
            with gr.Row():
                lora_out = gr.Textbox(label="Generated Robot Code", lines=8, interactive=False)
                lora_time = gr.Textbox(label="Time", lines=1, interactive=False)

        run.click(fn=demo.compare_models, inputs=[cmd],
                  outputs=[base_out, base_time, lora_out, lora_time])
        set_sample.click(fn=lambda s: s, inputs=[use_sample], outputs=[cmd])

    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True,
                    help="Path to your LoRA adapter folder (e.g., /home/ubuntu/Training-Robot/outputs/run-YYYYMMDD-HHMMSS)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    print("🚀 Starting Robot Model Comparison Demo...")
    app = create_interface(args.lora)
    # Set share=True if you want a public link
    app.launch(server_port=args.port, share=True)
