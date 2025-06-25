from dotenv import load_dotenv
from groq import Groq
import os

def init_groq_client():
    # You need to set your Groq API key here
    env_path = '.env'
    load_dotenv(dotenv_path=env_path)
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    #api_key = st.secrets.get("GROQ_API_KEY", "your_groq_api_key_here")
    return Groq(api_key=groq_api_key)

def call_groq_api(client, prompt, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    """Call Groq API with enhanced error handling and logging"""
    try:
        # log_step("Groq API Call", "started", f"Model: {model}")
        # logger.info(f"Prompt length: {len(prompt)} characters")
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4000
        )
        
        result = response.choices[0].message.content
        # log_step("Groq API Call", "success", f"Response length: {len(result)} characters")
        # logger.debug(f"API Response preview: {result[:200]}...")
        
        return result
    


    except Exception as e:
        # log_step("Groq API Call", "error", str(e))
        return str(e)
    

    # def call Azure_llm ()