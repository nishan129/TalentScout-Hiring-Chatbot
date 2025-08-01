
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Chatbot:
    """
    A class to generate questions for users based on their technical stack, experience level, and role type.
    """
    
    def __init__(self,api_key):
        """ Initialize the Chatbot.

        Args:
            api_key (str): API key for ChatGroq
        """
        self.api_key = api_key
        self.llm = ChatGroq(api_key=api_key, model="gemma2-9b-it")
        self.output_parser = StrOutputParser()
        

    
    def get_question(self, Answer, system_template):
        """Generate the question acording to user.

        Args:
            Answer (str): User answer the question.
            system_template (str): prompt for llm system to generate the questions.

        Raises:
            e: If any error in this code raise e

        Returns:
            str: Return questions.
        """
        try:
            prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system",system_template),
                        ("user","Answer:{Answer}")
                    ])
            
            chain = prompt | self.llm | self.output_parser
            
            logging.info("First Chat bot chain creation done ")
            
            question = chain.invoke({"Answer":Answer})
            
            logger.info(f"Successfully generated questions {question} ")
            
            return question
        except Exception as e:
            raise e