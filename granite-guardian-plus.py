import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- 1. CONFIGURATION & DEVICE ---
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.bfloat16
else:
    device = "cpu"
    dtype = torch.float32

print(f"--- Configuration ---")
print(f"Device: {device} | Dtype: {dtype}")

# --- 3. MODEL LOADING ---
INSTRUCT_ID = "ibm-granite/granite-3.3-8b-instruct"
GUARDIAN_ID = "ibm-granite/granite-guardian-3.3-8b"

print(f"Loading Instruct: {INSTRUCT_ID}...")
tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_ID)
model = AutoModelForCausalLM.from_pretrained(
    INSTRUCT_ID, torch_dtype=dtype, low_cpu_mem_usage=True
).to(device)

print(f"Loading Guardian: {GUARDIAN_ID}...")
guardian_tokenizer = AutoTokenizer.from_pretrained(GUARDIAN_ID)
guardian_model = AutoModelForCausalLM.from_pretrained(
    GUARDIAN_ID, torch_dtype=dtype, low_cpu_mem_usage=True
).to(device)

# --- 4. CHAT LOGIC ---
def chat():
    print("" + "="*60)
    print("  IBM GRANITE 3.3 + GRANITE GUARDIAN 3.3 (Official Templates)  ")
    print("  Mode: Thinking Enabled | Criteria: Harm Detection  ")
    print("  Type 'exit' to stop.  ")
    print("="*60 + "")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input or user_input.lower() in ["exit", "quit"]:
                break

            # --- STEP 1: GENERATION (INSTRUCT) ---
            messages = [{"role": "user", "content": user_input}]
            
            # Use apply_chat_template with thinking=True (Granite 3.3 specific)
            prompt = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False,
                thinking=True  # Enables <think>...<response> format
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            print("Granite is thinking...", end="")
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            )
            
            full_response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            # Extract response part if <response> tags are present
            if "<response>" in full_response:
                assistant_response = full_response.split("<response>")[1].split("</response>")[0].strip()
            else:
                assistant_response = full_response

            # --- STEP 2: MODERATION (GUARDIAN) ---
            # Evaluation of the assistant's response against 'harm' criteria
            guardian_messages = [{"role": "user", "content": assistant_response}]
            
            # Guardian template uses a specific guardian_config (or manual formatting)
            # Here we follow the judge pattern: evaluate if the assistant response is harmful
            guardian_prompt = guardian_tokenizer.apply_chat_template(
                guardian_messages,
                guardian_config={"criteria_id": "harm"},
                add_generation_prompt=True,
                tokenize=False
            )

            g_inputs = guardian_tokenizer(guardian_prompt, return_tensors="pt").to(device)
            
            g_outputs = guardian_model.generate(
                **g_inputs,
                max_new_tokens=128,
                temperature=0.1,
                do_sample=False
            )
            
            guardian_verdict = guardian_tokenizer.decode(g_outputs[0][g_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            # --- STEP 3: DISPLAY & DECISION ---
            # Guardian typically outputs <score>yes</score> or <score>no</score>
            is_harmful = "yes" in guardian_verdict.lower()

            if is_harmful:
                print(f"[!] SECURITY ALERT: Response blocked by Guardian.")
                print(f"Verdict: {guardian_verdict}")
            else:
                # Optional: display thinking if user wants, but here we show the response
                if "<think>" in full_response:
                    think_content = full_response.split("<think>")[1].split("</think>")[0].strip()
                    print(f"[Thought]: {think_content}")
                
                print(f"Granite: {assistant_response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")

    print("Goodbye!")

if __name__ == "__main__":
    chat()
