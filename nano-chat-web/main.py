import torch
import json
import asyncio
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import TextIteratorStreamer

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow Next.js (Vercel) frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading NanoRush Chat...")
# It downloads/caches Amogh1221/nano-chat locally on first run!
tokenizer = AutoTokenizer.from_pretrained("Amogh1221/nano-chat")
model = AutoModelForCausalLM.from_pretrained(
    "Amogh1221/nano-chat", 
    torch_dtype=torch.float16, 
    device_map="auto"
)

system_prompt = "You are NanoRush, an AI assistant created by Amogh Gupta. You are a helpful, respectful, and intelligent conversational partner. You must never pretend to be a human, and you must carefully pay attention to the conversation history."

class StopOnUser(StoppingCriteria):
    def __init__(self, prompt_length):
        self.prompt_length = prompt_length
    def __call__(self, input_ids, scores, **kwargs):
        generated_tokens = input_ids[0][self.prompt_length:]
        tail = tokenizer.decode(generated_tokens[-10:])
        return "\nUser:" in tail or "User:" in tail

@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    
    prompt = f"System: {system_prompt}\n\n"
    for msg in messages:
        if msg.get("role") == "user":
            prompt += f"User: {msg.get('content')}\n"
        elif msg.get("role") == "assistant" and msg.get("content") != "":
            prompt += f"Assistant: {msg.get('content')}\n"
    prompt += "Assistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    stop_criteria = StoppingCriteriaList([StopOnUser(prompt_length=inputs["input_ids"].shape[1])])
    
    # We use TextIteratorStreamer for web backends!
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
        prompt_lookup_num_tokens=3,
        stopping_criteria=stop_criteria,
        streamer=streamer,
    )
    
    # Run generation in a background thread so we can yield from the streamer
    import threading
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    async def event_generator():
        for text in streamer:
            # EventSourceResponse automatically formats strings into SSE events,
            # so we just yield the raw JSON string!
            yield json.dumps({'chunk': text})
            
            # Tiny sleep to allow the event loop to flush the chunk over network
            await asyncio.sleep(0.01)
            
    return EventSourceResponse(event_generator())
