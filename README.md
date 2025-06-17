# term.ai.persona.py

You can use this to run any AI Persona or system role with any LLama instruction model. Mileages will vary by success.

```
#!PYTHON_PATH/python
import torch
from transformers import pipeline
import torch
import re
import gradio as gr
from peft import PeftModel
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import TextStreamer
import sys

f = open(sys.argv[1],"r")
SYSTEM_PROMPT = f.read()


model_id = "NousResearch/Hermes-3-Llama-3.2-3B"


pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Custom stopping criteria to stop when the <|endoftext|> token is generated
class StopOnEndOfText(StoppingCriteria):
    def __init__(self, eos_token_id):
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if the last token generated is the eos_token_id
        return input_ids[0, -1] == self.eos_token_id

# Create an instance of the stopping criteria with the model's EOS token
eos_token_id = pipe.tokenizer.eos_token_id
stopping_criteria = StoppingCriteriaList([StopOnEndOfText(eos_token_id)])
textstreamer = TextStreamer(pipe.tokenizer, skip_prompt = True)

previous = []
def generate_text(system_role, user_input, sampling=True, temperature=0.9, top_p=0.9, top_k=20, alpha=0.9, max_length=8192, num_seqs=1):
    global previous
    messages = [
        {"role": "system", "content": system_role},        
    ]
    for i in previous:
        messages.append(i)
    messages.append({"role": "user", "content": user_input})

    outputs = pipe(
        messages,        
        streamer=textstreamer,
        do_sample=sampling,
        temperature = temperature,
        top_p = top_p,
        top_k = top_k,                
        max_length=max_length,
        num_return_sequences=num_seqs,        
        #remove_invalid_values=True,
        stopping_criteria=stopping_criteria,
        #note that these can mess it up very badly ... get bad tokenization and loco
        #repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        
    )
    output = outputs[0]["generated_text"][-1]['content']
    if(len(previous) > 3):
        previous = previous[1:]
    
    previous.append({"role": "user", "content": user_input})    
    return output 

while 1:
    print("Press CTRL+D to send.")
    p = sys.stdin.read()  
    output = generate_text(SYSTEM_PROMPT,p)
```
