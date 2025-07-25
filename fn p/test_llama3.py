from ollama import Client
import logging

logging.basicConfig(level=logging.INFO)

def test_llama3(prompt):
    try:
        client = Client(host='http://localhost:11434', timeout=60)  # Change to 11435 if needed
        response = client.chat(
            model='llama3',
            messages=[
                {'role': 'system', 'content': 'You are an AutoML assistant. Provide a clear and concise response.'},
                {'role': 'user', 'content': prompt}
            ]
        )
        result = response['message']['content']
        logging.info(f"LLaMA 3 Response: {result}")
        return result
    except Exception as e:
        logging.error(f"Error interacting with LLaMA 3: {e}")
        return None

def main():
    test_prompt = "Interpret this goal: 'Analyze a CSV to predict house prices.'"
    response = test_llama3(test_prompt)
    if response:
        print(f"Response: {response}")
    else:
        print("Failed to get response from LLaMA 3. Ensure Ollama server is running.")

if __name__ == "__main__":
    main()