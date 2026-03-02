from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os, json
import re

# Monkeypatch scaled_dot_product_attention to ignore 'enable_gqa' if it's not supported
_original_sdpa = F.scaled_dot_product_attention
def patched_sdpa(*args, **kwargs):
    kwargs.pop("enable_gqa", None)
    return _original_sdpa(*args, **kwargs)
F.scaled_dot_product_attention = patched_sdpa

app = Flask(__name__)

# Device configuration
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.bfloat16 # Better precision on Apple Silicon
else:
    device = "cpu"
    dtype = torch.float32

print(f"Using device: {device} with dtype: {dtype}")

# Shared Model ID for both Instruction and Moderation
model_id = "ibm-granite/granite-3.3-8b-instruct"
print(f"Loading shared model: {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=dtype,
    low_cpu_mem_usage=True
).to(device)

# Generator Pipeline
qa_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)

# Reuse the same model/tokenizer for moderation
guardian_tokenizer = tokenizer
guardian_model = model

@app.route('/')
def index():
    return render_template('chat.html', detector_response="") 

@app.route('/ask', methods=['POST'])
def ask():
    user_prompt = request.form['user_prompt']
    
    try:
        # Generate assistant response
        output = qa_pipe(
            user_prompt, 
            max_new_tokens=512, 
            temperature=0.8, 
            do_sample=True,
            truncation=True
        )
        assistant_response = output[0]['generated_text'].replace(user_prompt, "").strip()

        # Perform moderation check using a specific prompt for the instruct model
        moderation_prompt = f"""[INST] You are a safety classifier. Determine if the following response is harmful.
        
User Message: {user_prompt}
Assistant Message: {assistant_response}

Is the assistant message harmful? Answer with exactly 'Yes' or 'No'. [/INST]"""

        input_ids = guardian_tokenizer(moderation_prompt, return_tensors="pt").input_ids.to(guardian_model.device)
        guardian_model.eval()
        with torch.no_grad():
            output = guardian_model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=5,
                return_dict_in_generate=True,
            )
        
        moderation_text = guardian_tokenizer.decode(output.sequences[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
          
        if "yes" in moderation_text.lower():
            return jsonify({
                "status": "harmful", 
                "details": moderation_text, 
                "response": "Cette réponse a été bloquée car elle a été jugée potentiellement nocive."
            })
        elif "no" in moderation_text.lower():
            return jsonify({
                "status": "safe", 
                "response": assistant_response
            })
        else:
            return jsonify({
                "status": "unclear", 
                "details": moderation_text, 
                "response": assistant_response
            })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "status": "error",
            "response": f"An error occurred: {str(e)}"
        })

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)