import asyncio
import aiohttp
import json

DEEPSEEK_API_KEY = "your_deepseek_api_key_here"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

FLUX_API_KEY = "your_flux_api_key_here"
FLUX_API_URL = "https://api.bfl.ml/v1/flux-pro"

async def prompt_enhancement_agent(session, user_input: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    system_prompt = (
        "You are an expert prompt engineer for advanced text-to-image models like FLUX.1. "
        "Take the user's simple concept and expand it into a highly detailed, descriptive, "
        "and visually rich English prompt. Include lighting, camera angles, style, and mood. "
        "Return ONLY the English prompt."
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7
    }
    try:
        async with session.post(DEEPSEEK_API_URL, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return data['choices'][0]['message']['content'].strip()
    except Exception:
        return ""

async def image_generation_agent(session, english_prompt: str) -> str:
    if not english_prompt:
        return ""
    headers = {
        "Authorization": f"Bearer {FLUX_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": english_prompt,
        "width": 1024,
        "height": 768,
        "steps": 28,
        "prompt_upsampling": False
    }
    try:
        async with session.post(FLUX_API_URL, headers=headers, json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return data.get('image_url', '')
    except Exception:
        return ""

async def run_pipeline(user_inputs: list):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for user_input in user_inputs:
            enhanced_prompt = await prompt_enhancement_agent(session, user_input)
            if enhanced_prompt:
                task = asyncio.create_task(image_generation_agent(session, enhanced_prompt))
                tasks.append(task)
        return await asyncio.gather(*tasks)

if __name__ == "__main__":
    test_inputs = [
        "一位母亲在安慰打碎了瓷碗的孩子，温馨，治愈",
        "赛博朋克风格的未来城市，下雨天"
    ]
    results = asyncio.run(run_pipeline(test_inputs))
    print(results)