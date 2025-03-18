import wikipediaapi
import nltk
import logging
import os
import requests
from typing import Dict, List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikipediaCollector:
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent='AutoExam/1.0 (mnade@example.com)'
        )
        
        # Initialize NLTK data availability flag
        self.nltk_available = False
        try:
            nltk.data.find('tokenizers/punkt')
            self.nltk_available = True
            logger.info("NLTK data found and available")
        except LookupError:
            logger.info("NLTK data not found, using basic text processing")

    def get_topic_content(self, topic: str) -> Optional[str]:
        """Get content for a given topic from Wikipedia."""
        try:
            page = self.wiki.page(topic)
            if not page.exists():
                logger.warning(f"Topic '{topic}' not found on Wikipedia")
                return None
            return page.text
        except Exception as e:
            logger.error(f"Error fetching Wikipedia content: {str(e)}")
            return None

    def process_content(self, content: str) -> str:
        """Process the Wikipedia content for question generation."""
        if not content:
            return ""

        # Basic content cleaning
        lines = content.split('\n')
        processed_lines = []

        for line in lines:
            # Skip empty lines and references
            if not line.strip() or line.startswith('==') or '[' in line:
                continue
            processed_lines.append(line)

        processed_content = ' '.join(processed_lines)

        # Use NLTK for sentence tokenization if available
        if self.nltk_available:
            try:
                sentences = nltk.sent_tokenize(processed_content)
                # Keep only sentences that are likely to be informative
                sentences = [s for s in sentences if len(s.split()) > 5]
                return ' '.join(sentences)
            except Exception as e:
                logger.warning(f"NLTK processing failed: {str(e)}")

        # Fallback to basic processing
        return processed_content

# Initialize the collector
wikipedia_collector = WikipediaCollector() 