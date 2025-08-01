from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnswerBot:
    """
    This is Answer Bot generate the answer according to the questions.
    """
    
    def __init__(self,api_key,prompt):
        """Initialize the AnswerBot

        Args:
            api_key (str): ChatGroq api key
            prompt (str): System Prompt
        """
        self.api_key = api_key
        self.prompt = prompt
        self.llm = ChatGroq(api_key=api_key,model="gemma2-9b-it")
        self.output_parser = StrOutputParser()
        self.chat_prompt_template = ChatPromptTemplate(
            
                [
                        ("system",self.prompt),
                        ("user","Question:{Question}")
                    ])
    
    def answer(self, Question):
        """Creating the Answer acording to questions.

        Args:
            Question (str): Question generated by chatbot

        Raises:
            e: If any error in this code raise e

        Returns:
            str: Answer according to questions
        """
        try:
            logging.info("Answer bot chain  done ")
            chain = self.chat_prompt_template | self.llm | self.output_parser
            answer = chain.invoke({"Question":Question})
            logger.info(f"Successfully answer generated {answer} ")
            return answer
        except Exception as e:
            raise e