from src.utils.main_utils import get_all_user_message, get_all_ai_message, get_all_corect_message
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoreOptimizer:
    """
    A class to optimize and generate scores for user answers based on AI questions and correct answers.
    """

    def __init__(self, api_key: str, prompt: Dict[str, Any]):
        """
        Initialize the ScoreOptimizer.

        Args:
            api_key (str): API key for ChatGroq
            prompt (dict): Prompt configuration from YAML format
        """
        if not api_key:
            raise ValueError("API key cannot be empty")
        if not prompt or 'prompt_score' not in prompt:
            raise ValueError("Prompt configuration must contain 'prompt_score' key")

        self.api_key = api_key
        self.prompt = prompt
        self.llm = ChatGroq(api_key=self.api_key, model="gemma2-9b-it")
        self.output_parser = StrOutputParser()

    def _escape_prompt_template(self, prompt_template: str) -> str:
        """
        Escape curly braces in prompt template except for valid template variables.

        Args:
            prompt_template: The original prompt template

        Returns:
            str: Escaped prompt template
        """
        # Valid template variables that should NOT be escaped by LangChain's prompt
        # These are the variables that will be dynamically filled
        valid_template_variables = ['{question}', '{correct_answer}', '{user_answer}']

        # Use a more robust regex to find all unescaped single curly braces
        # that are not part of the `valid_template_variables`.

        # First, temporarily replace valid variables with unique placeholders
        # to prevent them from being escaped.
        placeholders = {}
        for i, var in enumerate(valid_template_variables):
            placeholder = f"__TEMP_VAR_{i}__"
            prompt_template = prompt_template.replace(var, placeholder)
            placeholders[placeholder] = var

        # Now, escape any remaining single curly braces by doubling them.
        # This regex looks for a single '{' not preceded by another '{'
        # and a single '}' not followed by another '}'.
        prompt_template = re.sub(r'(?<!\{)\{(?!\{)', '{{', prompt_template)
        prompt_template = re.sub(r'(?<!\})\}(?!\})', '}}', prompt_template)

        # Finally, restore the valid template variables from their placeholders.
        for placeholder, original_var in placeholders.items():
            prompt_template = prompt_template.replace(placeholder, original_var)

        return prompt_template

    def _create_scoring_prompt(self, question: str, correct_answer: str) -> ChatPromptTemplate:
        """
        Create a chat prompt template for scoring.

        Args:
            question: The question being scored
            correct_answer: The correct answer for the question

        Returns:
            ChatPromptTemplate: Configured prompt template
        """
        try:
            # Escape the prompt template first
            escaped_prompt_template = self._escape_prompt_template(self.prompt['prompt_score'])

            # Create the chat prompt template
            # The system message will contain the fixed parts of the prompt
            # The user message will contain the user's answer
            return ChatPromptTemplate.from_messages([
                ("system", escaped_prompt_template),
                ("user", "user_answer: {user_answer}")
            ])

        except KeyError as e:
            logger.error(f"Missing key in prompt template: {e}")
            raise ValueError(f"Prompt template is missing required key: {e}")
        except Exception as e:
            logger.error(f"Error creating scoring prompt: {e}")
            raise

    def _generate_single_score(self, question: str, correct_answer: str, user_answer: str) -> str:
        """
        Generate a score for a single question-answer pair.

        Args:
            question: The question
            correct_answer: The correct answer
            user_answer: The user's answer

        Returns:
            str: Generated score
        """
        try:
            chat_prompt_template = self._create_scoring_prompt(question, correct_answer)

            # Bind the values for 'question' and 'correct_answer' to the prompt template
            # and then invoke the chain with 'user_answer'.
            # The `partial` method allows you to "pre-fill" some of the template variables.
            chain = chat_prompt_template.partial(question=question, correct_answer=correct_answer) | self.llm | self.output_parser
            score = chain.invoke({"user_answer": user_answer})

            logger.info(f"Generated score for question: {question[:50]}...")
            logger.debug(f"Score: {score}")

            return score

        except Exception as e:
            logger.error(f"Error generating score for question: {question[:50]}... - {str(e)}")
            raise

    def generate_score(self, messages: Any) -> List[str]:
        """
        Generate scores for all question-answer pairs in the messages.

        Args:
            messages: Messages containing questions, correct answers, and user answers

        Returns:
            List[str]: List of generated scores

        Raises:
            Exception: If there's an error in processing or validation
        """
        try:
            # Extract messages using utility functions
            questions = get_all_ai_message(messages)
            correct_answers = get_all_corect_message(messages)
            user_answers = get_all_user_message(messages)

            # Log extracted data for debugging
            logger.info(f"Extracted {len(questions)} questions, {len(correct_answers)} correct answers, "
                        f"{len(user_answers)} user answers")

            # Check if there are any messages to process
            if not questions:
                logger.warning("No questions found in messages")
                return []

            scores = []

            # Process each question-answer pair
            for i, (question, correct_ans, user_ans) in enumerate(zip(questions, correct_answers, user_answers)):
                logger.info(f"Processing question {i + 1}/{len(questions)}")

                # Generate score for current pair
                score = self._generate_single_score(question, correct_ans, user_ans)
                scores.append(score)
                print(score)

            logger.info(f"Successfully generated {len(scores)} scores")
            return scores

        except Exception as e:
            logger.error(f"Error in generate_score: {str(e)}")
            raise