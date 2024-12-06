import requests

def chatgpt_api_call(prompt):
    api_key = 'key-here'

    url = 'https://api.openai.com/v1/chat/completions'
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    data = {
        'model': 'gpt-3.5-turbo',  # You can use 'gpt-4' if you have access
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'max_tokens': 100,  # Adjust this based on your needs
        'temperature': 0.7  # Adjust the creativity level of responses
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        return result['choices'][0]['message']['content']
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return None

# Example usage
if __name__ == "__main__":
    user_input = "What is the capital of France?"
    response = chatgpt_api_call(user_input)
    print("Response from ChatGPT:", response)

