import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sys

# Device configuration
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
print(f"Device: {device}")
print(f"Dtype: {dtype}\n")

# 1. Loading Instruction Model
instruct_id = "ibm-granite/granite-3.3-8b-instruct"
print(f"Loading Instruct model: {instruct_id}...")
tokenizer = AutoTokenizer.from_pretrained(instruct_id)
model = AutoModelForCausalLM.from_pretrained(
    instruct_id,
    torch_dtype=dtype,
    low_cpu_mem_usage=True
).to(device)

instruct_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

# 2. Loading Guardian Model
guardian_id = "ibm-granite/granite-guardian-3.3-8b"
print(f"Loading Guardian model: {guardian_id}...")
guardian_tokenizer = AutoTokenizer.from_pretrained(guardian_id)
guardian_model = AutoModelForCausalLM.from_pretrained(
    guardian_id,
    torch_dtype=dtype,
    low_cpu_mem_usage=True
).to(device)

guardian_pipe = pipeline(
    "text-generation",
    model=guardian_model,
    tokenizer=guardian_tokenizer,
    device=device
)

def chat():
    print("\n" + "="*50)
    print(" Granite 3.3 CLI + Granite Guardian 3.3 ")
    print(" Type 'exit' or 'quit' to stop. ")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break

            # --- STEP 1: GENERATION ---
            messages = [{"role": "user", "content": user_input}]
            
            output = instruct_pipe(
                messages, 
                max_new_tokens=512, 
                temperature=0.7, 
                do_sample=True,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end_of_text|>")]
            )
            
            assistant_response = output[0]['generated_text'][-1]['content'].strip()

            # --- STEP 2: MODERATION WITH GRANITE GUARDIAN ---
            # Format according to Granite Guardian documentation
            # role 'creator' defines the risk category
            # role 'user' contains the content to evaluate
            
            guardian_messages = [
                {"role": "creator", "content": "harm"},
                {"role": "user", "content": f"User Message: {user_input}\nAssistant Message: {assistant_response}"}
            ]
            
            # Using Non-Thinking mode (we want the score directly if possible)
            mod_output = guardian_pipe(
                guardian_messages,
                max_new_tokens=1024, # Large enough to allow <think> if the model wants
                temperature=0.1,
                do_sample=False
            )
            
            full_guardian_response = mod_output[0]['generated_text'][-1]['content'].strip()
            
            # Extract score from <score> tags if present, else look for yes/no
            if "<score>" in full_guardian_response:
                score = full_guardian_response.split("<score>")[1].split("</score>")[0].strip().lower()
            else:
                score = full_guardian_response.lower()

            # --- STEP 3: DECISION ---
            if "yes" in score:
                print(f"\n[BLOCKED] La réponse a été jugée nocive par Guardian.")
                print(f"(Raisonnement: {full_guardian_response})")
            else:
                print(f"\nGranite: {assistant_response}")
                # Optional: print thinking if available
                if "<think>" in full_guardian_response:
                    think = full_guardian_response.split("<think>")[1].split("</think>")[0].strip()
                    # print(f"\n[Guardian Thought]: {think}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

    print("\nGoodbye!")

if __name__ == "__main__":
    chat()
