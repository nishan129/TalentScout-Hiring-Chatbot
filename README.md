# 💼 TalentScout-Hiring-Chatbot
An intelligent Hiring Assistant chatbot developed for TalentScout, a fictional recruitment agency specializing in technology placements. The chatbot streamlines technical evaluations by asking domain-specific interview questions, evaluating candidate responses, and analyzing sentiment and proficiency in real-time.

# 🚀 Features
* Generate technical questions based on the candidate's tech stack.

* Real-time sentiment and confidence analysis of responses.

* Built using Streamlit, LangChain, LLM APIs, and more.

* Easily configurable and extensible for any technical domain.

# 🛠️ Setup Instructions
Follow these steps to set up and run the chatbot locally.

1. Clone the Repository

```
git clone https://github.com/nishan129/TalentScout-Hiring-Chatbot.git
cd TalentScout-Hiring-Chatbot
 ```

2. Create a Virtual Environment
You can use either uv, venv, or conda.

Option A: Using uv (recommended for fast installs)

```
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
 ```

Option B: Using conda

```
conda create -p venv/ python=3.10
conda activate venv/
 ```
3. Install Dependencies

```
pip install -r requirements.txt
 ```

4. Set up Environment Variables
Create a .env file in the project root and add your GROQ API key:

```
GROQ_API_KEY=your_api_key_here
 ```

5. Run the Streamlit App
```
streamlit run main.py
 ```

# 🧠 Technologies Used

* Streamlit – UI Framework for ML apps

* LangChain – LLM orchestration and prompt management

* GROQ API – LLM inference engine (can switch to OpenAI, Anthropic, etc.)

* Python – Core programming language

* dotenv – For managing environment variables

# 📁 Project Structure

```
TalentScout-Hiring-Chatbot/
├── main.py
├── src/
├── .env.example
├── requirements.txt
└── README.md
 ```

# 📌 TODO (Optional Enhancements)

* Add authentication and user session tracking

* Integrate with applicant tracking systems (ATS)

* Add support for voice input and feedback

* Save candidate session transcripts

# 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.