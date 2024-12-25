import logging
from src.config.settings import AZURE_CREDENTIALS
from src.data.data_loader import MovieDataLoader
from src.recommender.chatbot import MovieRecommendationBot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Loading and preprocessing movie data...")
        data_loader = MovieDataLoader()
        df, documents = data_loader.load_and_preprocess()
        
        logger.info("Initializing recommendation chatbot...")
        chatbot = MovieRecommendationBot(
            documents=documents,
            movie_data=df,
            azure_credentials=AZURE_CREDENTIALS
        )
        
        print("\nMovie Recommendation Chatbot")
        print("Type 'quit' to exit")
        print("="*50)
        
        while True:
            query = input("\nWhat kind of movie are you looking for? ")
            
            if query.lower() == 'quit':
                break
                
            try:
                response = chatbot.get_recommendation(query)
                print("\nRecommendations:")
                print(response)
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print("Sorry, I encountered an error. Please try a different query.")
                
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main()