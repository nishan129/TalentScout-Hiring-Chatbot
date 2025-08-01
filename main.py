import os
import streamlit as st
import json
from dotenv import load_dotenv
from src.bot.chat_bot import Chatbot
from src.analysis.sentiment_analysis import SentimentAnalysis
from src.utils.main_utils import get_last_assistant_message, get_last_user_message, read_yaml
from src.Optimize.scroe_optimizer import ScoreOptimizer
from src.answer_bot.bot import AnswerBot
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration and Initialization ---

# Page config
st.set_page_config(
    page_title="TalentScout Hiring Assistant",
    page_icon="🎯",
    layout="wide"
)

# Load environment variables
load_dotenv()

# Load prompts
@st.cache_data
def load_prompts():
    """Load prompts from YAML file with caching"""
    try:
        return read_yaml(file_path="src/prompts/prompt.yaml")
    except FileNotFoundError:
        st.error("Error: prompt.yaml not found. Please ensure it's in the 'src/prompts' directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading prompts: {str(e)}")
        st.stop()

PROMPTS = load_prompts()

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        font-size: 2.5rem;
        margin-bottom: 2rem;
    }
    .section-header {
        color: #1B4F72;
        font-size: 1.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498DB;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #1B4F72;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498DB;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D4EDDA;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #27AE60;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #F8D7DA;
        color: #721C24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #DC3545;
        margin: 1rem 0;
    }
    .score-box {
        background-color: #FFF3CD;
        color: #856404;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FFC107;
        margin: 1rem 0;
        text-align: center;
    }
    .question-counter {
        background-color: #E8F4FD;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
        color: #1B4F72;
    }
    .completion-box {
        background-color: #D1ECF1;
        color: #0C5460;
        padding: 2rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17A2B8;
        margin: 2rem 0;
        text-align: center;
    }
    .answer-box {
        background-color: #57329F;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4169E1;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
def initialize_session_state():
    """Initialize all session state variables"""
    default_values = {
        'candidate_data': {},
        'form_submitted': False,
        'chat_started': False,
        'current_question': 0,
        'max_questions': 3,
        'interview_completed': False,
        'show_score': False,
        'waiting_for_answer': False,
        'messages': [],
        'processing_answer': False,
        'error_occurred': False,
        'last_error': None
    }
    
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Helper Functions ---

def reset_interview_state():
    """Resets all session state variables related to the interview."""
    logger.info("Resetting interview state")
    interview_keys = [
        'chat_started', 'current_question', 'interview_completed', 
        'show_score', 'waiting_for_answer', 'messages', 
        'processing_answer', 'error_occurred', 'last_error'
    ]
    
    for key in interview_keys:
        if key == 'messages':
            st.session_state[key] = []
        elif key in ['current_question']:
            st.session_state[key] = 0
        else:
            st.session_state[key] = False

def get_experience_level(years_of_experience):
    """Maps years of experience to a general experience level."""
    exp_mapping = {
        "0-1 years": "Junior",
        "1-2 years": "Junior", 
        "2-3 years": "Junior",
        "3-5 years": "Mid",
        "5-7 years": "Mid",
        "7-10 years": "Senior",
        "10+ years": "Senior"
    }
    return exp_mapping.get(years_of_experience, "Mid")

def validate_required_fields(candidate_data):
    """Validate that all required fields are filled"""
    required_fields = ['full_name', 'email', 'phone', 'location']
    missing_fields = [field for field in required_fields if not candidate_data.get(field)]
    
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    
    if candidate_data.get('experience_years') == "Select...":
        return False, "Please select years of experience"
    
    if not candidate_data.get('desired_positions'):
        return False, "Please select at least one desired position"
        
    if not candidate_data.get('tech_stack'):
        return False, "Please select your technology stack"
        
    if not candidate_data.get('key_technologies'):
        return False, "Please select your key technologies"
    
    return True, ""

def display_error(error_message, show_retry=False):
    """Display error message with optional retry button"""
    st.markdown(f"""
    <div class="error-box">
        <h4>⚠️ Error Occurred</h4>
        <p>{error_message}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if show_retry:
        if st.button("🔄 Retry", key="retry_button"):
            st.session_state.error_occurred = False
            st.session_state.last_error = None
            st.rerun()

def process_user_answer(user_input, system_template, model, answer_bot, analysis):
    """Process user answer and generate next question or complete interview"""
    try:
        logger.info(f"Processing user answer for question {st.session_state.current_question + 1}")
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get the last assistant question
        last_question = get_last_assistant_message(st.session_state.messages)
        if not last_question:
            raise ValueError("Could not retrieve the last question")
        
        logger.info(f"Last question retrieved: {last_question[:100]}...")
        
        # Generate correct answer using answer_bot
        with st.status("Generating correct answer...", expanded=False) as status:
            try:
                correct_answer = answer_bot.answer(Question=last_question)
                if not correct_answer:
                    raise ValueError("Answer bot returned empty response")
                
                logger.info(f"Correct answer generated: {correct_answer[:100]}...")
                
                # Store correct answer in messages for scoring
                st.session_state.messages.append({
                    "role": "correct_answer", 
                    "content": correct_answer,
                    "question_number": st.session_state.current_question + 1
                })
                status.update(label="✅ Correct answer generated", state="complete")
                
            except Exception as e:
                logger.error(f"Error generating correct answer: {str(e)}")
                status.update(label="❌ Error generating correct answer", state="error")
                # Continue without correct answer - we can still proceed with analysis
                correct_answer = "Could not generate correct answer"
        
        # Perform sentiment analysis
        with st.status("Analyzing your response...", expanded=False) as status:
            try:
                human_message = get_last_user_message(st.session_state.messages)
                analysis_result = analysis.analysis(
                    human_message=human_message, 
                    ai_message=last_question
                )
                
                if not analysis_result:
                    analysis_result = "Analysis completed successfully"
                
                logger.info(f"Analysis completed: {analysis_result[:100]}...")
                status.update(label="✅ Response analyzed", state="complete")
                
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {str(e)}")
                analysis_result = f"Analysis error: {str(e)}"
                status.update(label="⚠️ Analysis completed with issues", state="error")
        
        # Increment question counter
        st.session_state.current_question += 1
        
        # Determine next action
        if st.session_state.current_question < st.session_state.max_questions:
            # Generate next question
            with st.status("Preparing next question...", expanded=False) as status:
                try:
                    question_prompt = (
                        f"Generate question {st.session_state.current_question + 1} of "
                        f"{st.session_state.max_questions}. Make it different from previous "
                        f"questions and relevant to the candidate's profile and previous answers."
                    )
                    
                    next_question = model.get_question(
                        system_template=system_template, 
                        Answer=question_prompt
                    )
                    
                    if not next_question:
                        raise ValueError("Model returned empty question")
                    
                    logger.info(f"Next question generated: {next_question[:100]}...")
                    
                    # Add analysis and next question to messages
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"**Analysis:** {analysis_result}\n\n**Next Question:** {next_question}",
                        "question_number": st.session_state.current_question + 1
                    })
                    
                    st.session_state.waiting_for_answer = True
                    status.update(label="✅ Next question ready", state="complete")
                    
                except Exception as e:
                    logger.error(f"Error generating next question: {str(e)}")
                    status.update(label="❌ Error generating next question", state="error")
                    raise e
        else:
            # Interview completed
            logger.info("Interview completed - all questions answered")
            completion_message = (
                f"🎉 Congratulations! You have successfully completed all "
                f"{st.session_state.max_questions} questions. Your interview is now complete."
            )
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"**Final Analysis:** {analysis_result}\n\n**Status:** {completion_message}",
                "is_completion": True
            })
            
            st.session_state.interview_completed = True
            st.session_state.waiting_for_answer = False
        
        return True, "Success"
        
    except Exception as e:
        error_msg = f"Error processing answer: {str(e)}"
        logger.error(error_msg)
        st.session_state.error_occurred = True
        st.session_state.last_error = error_msg
        return False, error_msg

# --- Main Application Logic ---

st.markdown('<h1 class="main-header">🎯 TalentScout Hiring Assistant</h1>', unsafe_allow_html=True)

# Sidebar for candidate information form
with st.sidebar:
    st.markdown('<h2 class="section-header">📋 Candidate Information</h2>', unsafe_allow_html=True)

    with st.form("candidate_form"):
        # Personal Information
        st.subheader("Personal Details")
        full_name = st.text_input("Full Name *", placeholder="Enter your full name")
        email = st.text_input("Email Address *", placeholder="your.email@example.com")
        phone = st.text_input("Phone Number *", placeholder="+1 (555) 123-4567")
        location = st.text_input("Current Location *", placeholder="City, State/Country")

        # Professional Information
        st.subheader("Professional Details")
        experience_years = st.selectbox(
            "Years of Experience *",
            ["Select...", "0-1 years", "1-2 years", "2-3 years", "3-5 years", "5-7 years", "7-10 years", "10+ years"]
        )

        desired_positions = st.multiselect(
            "Desired Position(s) *",
            ["Software Engineer", "Data Scientist", "ML Engineer", "AI Researcher", "Full Stack Developer",
             "Backend Developer", "Frontend Developer", "DevOps Engineer", "Product Manager", "Tech Lead", "Other"],
            help="Select one or more positions you're interested in"
        )

        other_position = ""
        if "Other" in desired_positions:
            other_position = st.text_input("Specify Other Position(s)", placeholder="e.g., Blockchain Developer")

        # Technical Skills
        st.subheader("Technical Skills")
        tech_stack = st.multiselect(
            "Technology Stack *",
            ["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "Ruby", "PHP", "Swift", "Kotlin",
             "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring", "Laravel", ".NET", "Other"],
            help="Select your primary programming languages and frameworks"
        )

        other_tech = ""
        if "Other" in tech_stack:
            other_tech = st.text_input("Specify Other Technologies", placeholder="e.g., Scala, Elixir")

        key_technologies = st.multiselect(
            "Key Technologies/Specializations *",
            ["Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision",
             "Data Science", "Big Data", "Cloud Computing (AWS)", "Cloud Computing (Azure)",
             "Cloud Computing (GCP)", "DevOps", "Microservices", "Blockchain", "IoT", "Cybersecurity",
             "Mobile Development", "Web Development", "Database Management", "API Development", "Other"],
            help="Select your areas of expertise"
        )

        other_key_tech = ""
        if "Other" in key_technologies:
            other_key_tech = st.text_input("Specify Other Key Technologies", placeholder="e.g., Quantum Computing")

        # Submit button
        submitted = st.form_submit_button("Start Interview Process", use_container_width=True)

        if submitted:
            # Prepare candidate data
            candidate_data = {
                "full_name": full_name,
                "email": email,
                "phone": phone,
                "location": location,
                "experience_years": experience_years,
                "desired_positions": desired_positions + ([other_position] if "Other" in desired_positions and other_position else []),
                "tech_stack": tech_stack + ([other_tech] if "Other" in tech_stack and other_tech else []),
                "key_technologies": key_technologies + ([other_key_tech] if "Other" in key_technologies and other_key_tech else [])
            }
            
            # Validation
            is_valid, error_message = validate_required_fields(candidate_data)
            
            if not is_valid:
                st.error(error_message)
            else:
                # Store candidate data and reset interview state
                st.session_state.candidate_data = candidate_data
                st.session_state.form_submitted = True
                reset_interview_state()
                st.success("✅Your Data is not stored and shared. All interaction are process localy and delete when you close the browser. Start Inteview")
                st.rerun()

    # Show interview progress in sidebar if started
    if st.session_state.chat_started and not st.session_state.interview_completed:
        st.markdown("---")
        st.markdown("### 📊 Interview Progress")
        progress = st.session_state.current_question / st.session_state.max_questions
        st.progress(progress)
        st.markdown(f"**Questions Completed:** {st.session_state.current_question}/{st.session_state.max_questions}")
        
        # Show current status
        if st.session_state.waiting_for_answer:
            st.info("⏳ Waiting for your answer...")
        elif st.session_state.processing_answer:
            st.info("🔄 Processing your response...")

# Main content area
if not st.session_state.form_submitted:
    st.markdown("""
    <div class="info-box">
        <h3>Welcome to TalentScout! 👋</h3>
        <p>Please fill out the candidate information form in the sidebar to get started with your technical interview process.</p>
        <p>We'll ask you exactly <strong>3 technical questions</strong> and then provide your performance score.</p>
    </div>
    """, unsafe_allow_html=True)

    # Display what we offer
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🎯 What We Offer")
        st.markdown("""
        - **Exactly 3 Questions**: Focused technical assessment
        - Experience-level appropriate difficulty
        - Real-time interview simulation
        - Comprehensive performance scoring
        - Detailed feedback and analysis
        """)

    with col2:
        st.markdown("### 🚀 Interview Process")
        st.markdown("""
        1. **Fill Information**: Complete the form in the sidebar
        2. **Answer Question 1**: First technical question
        3. **Answer Question 2**: Second technical question
        4. **Answer Question 3**: Final technical question
        5. **Get Your Score**: Detailed performance analysis
        """)

else:
    # Display candidate summary
    st.markdown('<h2 class="section-header">👤 Candidate Profile</h2>', unsafe_allow_html=True)

    candidate = st.session_state.candidate_data
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        **Name:** {candidate['full_name']}  
        **Email:** {candidate['email']}  
        **Phone:** {candidate['phone']}  
        **Location:** {candidate['location']}
        """)

    with col2:
        st.markdown(f"""
        **Experience:** {candidate['experience_years']}  
        **Desired Roles:** {', '.join(candidate['desired_positions'])}
        """)

    with col3:
        # Display only first few to prevent overflow
        tech_stack_display = ', '.join(candidate['tech_stack'][:3])
        if len(candidate['tech_stack']) > 3:
            tech_stack_display += '...'

        key_tech_display = ', '.join(candidate['key_technologies'][:2])
        if len(candidate['key_technologies']) > 2:
            key_tech_display += '...'

        st.markdown(f"""
        **Tech Stack:** {tech_stack_display}  
        **Specializations:** {key_tech_display}
        """)

    # Interview Section
    st.markdown('<h2 class="section-header">💬 Technical Interview</h2>', unsafe_allow_html=True)

    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("🔑 GROQ_API_KEY not found in environment variables. Please check your .env file.")
        st.info("💡 You can get your API key from [Groq Console](https://console.groq.com/keys).")
        st.stop()

    # Initialize AI models
    try:
        # Initialize models with error handling
        model = Chatbot(api_key=api_key)
        
        if 'answer_bot' not in PROMPTS:
            st.error("❌ 'answer_bot' prompt not found in PROMPTS configuration")
            st.stop()
            
        answer_bot = AnswerBot(api_key=api_key, prompt=PROMPTS['answer_bot'])
        
        if 'prompt_analysis' not in PROMPTS:
            st.error("❌ 'prompt_analysis' prompt not found in PROMPTS configuration")
            st.stop()
            
        analysis = SentimentAnalysis(api_key=api_key, prompt=PROMPTS['prompt_analysis'])
        score_optimizer = ScoreOptimizer(api_key=api_key, prompt=PROMPTS)

        # Create system template based on candidate data
        experience_level = get_experience_level(candidate['experience_years'])
        
        if 'prompt_bot' not in PROMPTS:
            st.error("❌ 'prompt_bot' prompt not found in PROMPTS configuration")
            st.stop()
            
        system_template = PROMPTS['prompt_bot'].format(
            experience_level=experience_level,
            experience_years=candidate['experience_years'],
            desired_positions=candidate['desired_positions'],
            tech_stack=candidate['tech_stack'],
            key_technologies=candidate['key_technologies']
        )

        # Handle error states
        if st.session_state.error_occurred:
            display_error(st.session_state.last_error, show_retry=True)

        # Interview States Management
        elif not st.session_state.chat_started and not st.session_state.interview_completed:
            # Start interview button
            if st.button("🚀 Start 3-Question Technical Interview", use_container_width=True):
                logger.info("Starting new interview")
                st.session_state.chat_started = True
                st.session_state.current_question = 0
                st.session_state.messages = []
                st.session_state.interview_completed = False
                st.session_state.show_score = False
                st.session_state.waiting_for_answer = False
                st.session_state.processing_answer = False
                st.session_state.error_occurred = False
                st.rerun()

        elif st.session_state.interview_completed and not st.session_state.show_score:
            # Interview completed - show completion message and score option
            st.markdown(f"""
            <div class="completion-box">
                <h3>🎉 Interview Completed Successfully!</h3>
                <p><strong>Congratulations, {candidate['full_name']}!</strong></p>
                <p>You have successfully answered all <strong>{st.session_state.max_questions} technical questions</strong>.</p>
                <p>Click below to see your detailed performance analysis and score.</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("📊 Check My Interview Score", use_container_width=True):
                st.session_state.show_score = True
                st.rerun()

        elif st.session_state.show_score:
            # Show score analysis
            st.markdown(f"""
            <div class="score-box">
                <h3>📊 Your Interview Performance Analysis</h3>
                <p>Based on your answers to all {st.session_state.max_questions} questions</p>
            </div>
            """, unsafe_allow_html=True)

            try:
                # Generate comprehensive score analysis
                with st.spinner("🔄 Analyzing your interview performance..."):
                    # Filter messages for scoring (only user and assistant messages)
                    conversation_history = [
                        msg for msg in st.session_state.messages
                        if msg["role"] in ["user", "assistant", "correct_answer"]
                    ]
                    
                    if conversation_history:
                        score_results = score_optimizer.generate_score(conversation_history)
                        
                        st.markdown("### 🎯 Detailed Score Analysis:")
                        
                        if isinstance(score_results, list):
                            total_overall_score = 0
                            valid_scores = 0
                            
                            for i, score_data in enumerate(score_results, 1):
                                st.markdown(f"#### Question {i} Performance:")
                                
                                try:
                                    # Try to parse JSON if it's a string
                                    if isinstance(score_data, str):
                                        import json
                                        # Extract JSON from the response if it contains other text
                                        start_idx = score_data.find('{')
                                        end_idx = score_data.rfind('}') + 1
                                        if start_idx != -1 and end_idx != 0:
                                            json_str = score_data[start_idx:end_idx]
                                            parsed_score = json.loads(json_str)
                                        else:
                                            # If no JSON found, display raw text
                                            st.markdown(score_data)
                                            continue
                                    else:
                                        parsed_score = score_data
                                    
                                    # Display scores in a structured format
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric("Relevance", f"{parsed_score.get('relevance_score', 0)}/10")
                                        st.metric("Accuracy", f"{parsed_score.get('accuracy_score', 0)}/10")
                                    
                                    with col2:
                                        st.metric("Completeness", f"{parsed_score.get('completeness_score', 0)}/10")
                                        st.metric("Clarity", f"{parsed_score.get('clarity_score', 0)}/10")
                                    
                                    with col3:
                                        st.metric("Depth", f"{parsed_score.get('depth_score', 0)}/10")
                                        overall_score = parsed_score.get('overall_score', 0)
                                        st.metric("Overall Score", f"{overall_score:.1f}/10", delta=None)
                                    
                                    total_overall_score += overall_score
                                    valid_scores += 1
                                    
                                except (json.JSONDecodeError, KeyError, TypeError) as e:
                                    # If parsing fails, display as raw text
                                    st.markdown(f"**Raw Analysis:** {score_data}")
                                    logger.warning(f"Could not parse score JSON for question {i}: {e}")
                                
                                st.markdown("---")
                            
                            # Display overall interview performance
                            if valid_scores > 0:
                                average_score = total_overall_score / valid_scores
                                st.markdown("### 🏆 Overall Interview Performance")
                                
                                # Color code the overall score
                                if average_score >= 8:
                                    score_color = "#27AE60"  # Green
                                    performance_level = "Excellent"
                                elif average_score >= 6:
                                    score_color = "#F39C12"  # Orange
                                    performance_level = "Good"
                                elif average_score >= 4:
                                    score_color = "#E67E22"  # Dark Orange
                                    performance_level = "Fair"
                                else:
                                    score_color = "#E74C3C"  # Red
                                    performance_level = "Needs Improvement"
                                
                                st.markdown(f"""
                                <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {score_color}20; border: 2px solid {score_color};">
                                    <h2 style="color: {score_color}; margin: 0;">Average Score: {average_score:.1f}/10</h2>
                                    <h3 style="color: {score_color}; margin: 10px 0;">Performance Level: {performance_level}</h3>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown(score_results)
                    else:
                        st.warning("⚠️ No conversation history found for scoring.")

                # Show correct answers if available
                correct_answers = [
                    msg for msg in st.session_state.messages 
                    if msg["role"] == "correct_answer"
                ]
                
                if correct_answers:
                    with st.expander("🔍 View Correct Answers", expanded=False):
                        for i, answer_msg in enumerate(correct_answers, 1):
                            st.markdown(f"""
                            <div class="answer-box">
                                <h4>Question {answer_msg.get('question_number', i)} - Correct Answer:</h4>
                                <p>{answer_msg['content']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                # Option to restart interview
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Take Another Interview", use_container_width=True):
                        reset_interview_state()
                        st.rerun()
                
                with col2:
                    if st.button("👤 Update Profile", use_container_width=True):
                        st.session_state.form_submitted = False
                        reset_interview_state()
                        st.rerun()
                        
            except Exception as e:
                error_msg = f"Error generating score: {str(e)}"
                logger.error(error_msg)
                display_error(error_msg, show_retry=True)

        # Active Interview Flow
        elif st.session_state.chat_started and not st.session_state.interview_completed:

            # Show current question number
            st.markdown(f"""
            <div class="question-counter">
                Question {st.session_state.current_question + 1} of {st.session_state.max_questions}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="success-box">
                <strong>Interview in Progress</strong> - 
                Good luck, {candidate['full_name']}! Answer each question thoughtfully.
            </div>
            """, unsafe_allow_html=True)

            # Display all messages
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.chat_message("user").markdown(message["content"])
                elif message["role"] == "assistant":
                    # Handle different types of assistant messages
                    content = message["content"]
                    if "**Analysis:**" in content and "**Next Question:**" in content:
                        # Split analysis and question
                        parts = content.split("**Next Question:**")
                        if len(parts) == 2:
                            st.chat_message("assistant").markdown(parts[0])  # Analysis
                            st.chat_message("assistant").markdown(f"**Next Question:**{parts[1]}")  # Question
                        else:
                            st.chat_message("assistant").markdown(content)
                    elif "**Final Analysis:**" in content and "**Status:**" in content:
                        # Split final analysis and completion status
                        parts = content.split("**Status:**")
                        if len(parts) == 2:
                            st.chat_message("assistant").markdown(parts[0])  # Final Analysis
                            st.chat_message("assistant").markdown(f"**Status:**{parts[1]}")  # Status
                        else:
                            st.chat_message("assistant").markdown(content)
                    else:
                        st.chat_message("assistant").markdown(content)

            # Generate first question if no messages exist
            if not st.session_state.messages and not st.session_state.waiting_for_answer:
                with st.spinner("🤖 Generating your first question..."):
                    try:
                        question_prompt = f"Generate question 1 of {st.session_state.max_questions} technical interview questions."
                        first_question = model.get_question(system_template=system_template, Answer=question_prompt)
                        
                        if not first_question:
                            raise ValueError("Model returned empty first question")
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": first_question,
                            "question_number": 1
                        })
                        st.session_state.waiting_for_answer = True
                        logger.info("First question generated successfully")
                        st.rerun()
                        
                    except Exception as e:
                        error_msg = f"Error generating first question: {str(e)}"
                        logger.error(error_msg)
                        st.session_state.error_occurred = True
                        st.session_state.last_error = error_msg
                        st.rerun()

            # Chat input - only show if not completed and waiting for an answer
            if (st.session_state.current_question < st.session_state.max_questions and 
                st.session_state.waiting_for_answer and 
                not st.session_state.processing_answer):
                
                user_input = st.chat_input(f"Your answer to Question {st.session_state.current_question + 1}...")
                
                if user_input:
                    st.session_state.processing_answer = True
                    st.session_state.waiting_for_answer = False
                    
                    # Process the answer in a separate function
                    success, message = process_user_answer(user_input, system_template, model, answer_bot, analysis)
                    
                    st.session_state.processing_answer = False
                    
                    if not success:
                        st.session_state.error_occurred = True
                        st.session_state.last_error = message
                    
                    st.rerun()

    except Exception as e:
        error_msg = f"Error initializing AI models: {str(e)}"