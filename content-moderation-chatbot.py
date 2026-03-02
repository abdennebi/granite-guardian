from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os, json
import re

# Monkeypatch scaled_dot_product_attention to ignore 'enable_gqa' if it's not supported
# _original_sdpa = F.scaled_dot_product_attention
# def patched_sdpa(*args, **kwargs):
#    kwargs.pop("enable_gqa", None)
#    return _original_sdpa(*args, **kwargs)
# F.scaled_dot_product_attention = patched_sdpa

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
    torch_dtype=dtype,
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
        # Define the conversation
        messages = [
            {"role": "user", "content": user_prompt}
        ]

        # Generate assistant response using the pipeline with messages
        # This is more robust as the pipeline handles the template internally
        output = qa_pipe(
            messages, 
            max_new_tokens=512, 
            temperature=0.7, 
            do_sample=True,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end_of_text|>")]
        )
        
        # The output format for chat pipeline is different: [{'generated_text': [...]}]
        assistant_response = output[0]['generated_text'][-1]['content'].strip()

        # Perform moderation check using the same logic
        moderation_messages = [
            {"role": "system", "content": "You are a safety classifier. Determine if the following response is harmful. Answer with exactly 'Yes' or 'No'."},
            {"role": "user", "content": f"User Message: {user_prompt}\nAssistant Message: {assistant_response}\n\nIs the assistant message harmful?"}
        ]
        
        # We can use the pipeline here too for consistency
        mod_output = qa_pipe(
            moderation_messages,
            max_new_tokens=10,
            temperature=0.1, # Lower temperature for classification
            do_sample=False,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end_of_text|>")]
        )
        
        moderation_text = mod_output[0]['generated_text'][-1]['content'].strip()
          
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