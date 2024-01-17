import os
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
model = PeftModel.from_pretrained(base_model, 'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora')
model = model.eval()

news_dir = 'stock-movement-predictor/fin_data/results_with_content_clean'
stock_dir = 'stock-movement-predictor/fin_data/stock_data'


m2m_model_name = 'facebook/m2m100_418M'
m2m_tokenizer = M2M100Tokenizer.from_pretrained(m2m_model_name)
m2m_model = M2M100ForConditionalGeneration.from_pretrained(m2m_model_name)

def translate(text, src_lang="zh", target_lang="en"):
    m2m_tokenizer.src_lang = src_lang
    encoded_zh = m2m_tokenizer(text, return_tensors="pt")
    generated_tokens = m2m_model.generate(**encoded_zh, forced_bos_token_id=m2m_tokenizer.get_lang_id(target_lang))
    return m2m_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
SYSTEM_PROMPT = """You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n"""

def generate_model_prompt(news_df, stock_df, ticker):
    recent_stock = stock_df.sort_values(by='time', ascending=False).iloc[0]

    stock_movement = f"From {stock_df['time'].min()} to {recent_stock['time']}, {ticker}'s stock price moved from {stock_df['open'].iloc[0]} to {recent_stock['close']}."
    translated_news = [translate(f"{row['title']} {row['content'][:200]}...") for _, row in news_df.iterrows()]

    news_summary = '\n'.join([f"[Content]: {news_item}" for news_item in translated_news])
    prompt = f"""
    [Company Introduction]:
    {ticker} has shown notable movements in the stock market recently. 
    {stock_movement}
    Company news during this period are listed below:
    {news_summary}
    Based on all the information before {recent_stock['time']}, let's first analyze the positive developments and potential concerns for {ticker}. 
    """
    prompt += B_SYS + SYSTEM_PROMPT + E_SYS + E_INST
    return prompt

def process_stock_data(ticker):
    news_file = os.path.join(news_dir, ticker + '.csv')
    try:
        news_df = pd.read_csv(news_file) if os.path.exists(news_file) else pd.DataFrame()
    except Exception as e:
        print(f"Error reading {news_file}: {e}")
        news_df = pd.DataFrame()

    stock_file = os.path.join(stock_dir, ticker + '.csv')
    try:
        stock_df = pd.read_csv(stock_file) if os.path.exists(stock_file) else pd.DataFrame()
    except Exception as e:
        print(f"Error reading {stock_file}: {e}")
        stock_df = pd.DataFrame()

    return news_df, stock_df

output_file_path = 'forecasts_output.txt'

with open(output_file_path, 'a') as file:
    for file_name in os.listdir(stock_dir):
        if file_name.endswith('.csv'):
            ticker = file_name.split('.')[0]
            news_df, stock_df = process_stock_data(ticker)

            if news_df.empty or stock_df.empty:
                continue

            prompt = generate_model_prompt(news_df, stock_df, ticker)
            inputs = tokenizer(prompt, return_tensors='pt')
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            try:
                res = model.generate(**inputs, max_length=4096, do_sample=True, eos_token_id=tokenizer.eos_token_id, use_cache=True)
                output = tokenizer.decode(res[0], skip_special_tokens=True)
                answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)

                file.write(f"Forecast for {ticker}:\n")
                file.write(answer)
                file.write("\n" + "="*50 + "\n\n")
            except Exception as e:
                print(f"Error generating forecast for {ticker}: {e}")
