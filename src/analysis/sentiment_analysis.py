from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalysis:
    """
    A class to analyze users' answers based on the Sentiment expressed in their responses to given questions.
    """
    def __init__(self, api_key, prompt):
        """Initialize the SentimentAnalysis

        Args:
            api_key (str): ChatGroq api key
            prompt (str): System prompt for llm
        """
        
        self.api_key = api_key
        self.prompt = ChatPromptTemplate(
            [
                 ("system",prompt),
                        ("user","Answer:{Answer}")
            ]
        )
        self.llm = ChatGroq(api_key=api_key, model="gemma2-9b-it")
        self.output_parser = StrOutputParser()
    
    def analysis(self,human_message, ai_message):
        """Analysis the user sentiment

        Args:
            human_message (str): answer to the questions 
            ai_message (str): question generate by Chatbot

        Raises:
            e: If any error in this code raise e

        Returns:
            str: Retrun user more confident or some user confused.
        """
        try:
            logging.info("Analysis bot chain creation done ")
            chain = self.prompt | self.llm | self.output_parser
            message =        [ AIMessage(content=ai_message),
                                HumanMessage(content=human_message)]
            analysis = chain.invoke(message)
            logger.info(f"Successfully analysis user sentiment {analysis} ")
            
            return analysis
    
        except Exception as e:
            raise e