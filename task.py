from flask import Flask, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_KEY)


conversation_history = [
    {"role": "system", "content": (
        "You are a friendly and motivating Fitness Coach. "
        "Give clear, practical advice about workouts, diet, lifestyle, and motivation. "
        "Keep answers short, clean, bulleted and easy to follow. "
        "IMPORTANT: Only answer fitness, diet, workout, or health-related questions. "
        "If the user asks about unrelated topics (AI, coding, politics, etc.), "
        "politely refuse and redirect them back to fitness/health topics."
    )}
]



@app.route('/ask', methods=['POST'])
def ask_question():
    user_input = request.data.decode('utf-8').strip()

    if not user_input:
        return jsonify({"error": "No question provided"}), 400

    conversation_history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )

    answer = response.choices[0].message.content.strip()
    answer = " ".join(answer.split())

    conversation_history.append({"role": "assistant", "content": answer})

    return jsonify({"question": user_input, "answer": answer})

@app.route('/')
def home():
    return "Fitness Coach Chatbot is running! Use POST request or /ask to chat."

if __name__ == '__main__':
    app.run(debug=True)
