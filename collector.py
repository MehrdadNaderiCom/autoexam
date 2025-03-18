import wikipedia
import logging
import nltk
import os
import requests
from typing import Dict, List, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikipediaCollector:
    def __init__(self):
        """Initialize the WikipediaCollector."""
        logger.info("Initializing WikipediaCollector")
        wikipedia.set_lang('en')
        self.use_nltk = True
        try:
            nltk.data.find('tokenizers/punkt')
            logger.info("NLTK data found")
        except LookupError:
            logger.info("NLTK data not found, using basic text processing")
            self.use_nltk = False

    def get_topic_content(self, topic: str) -> Optional[Dict[str, str]]:
        """Get content from Wikipedia for a given topic."""
        try:
            logger.info(f"Searching Wikipedia for topic: {topic}")
            # Search for the topic
            search_results = wikipedia.search(topic)
            if not search_results:
                logger.warning(f"No Wikipedia results found for topic: {topic}")
                return None

            # Try each search result until we find a valid page
            for page_title in search_results[:3]:  # Try first 3 results
                try:
                    page = wikipedia.page(page_title, auto_suggest=False)
                    content = page.content
                    if not content:
                        continue

                    # Process the content
                    processed_content = self.process_content(content)
                    if not processed_content:
                        continue

                    return {
                        'text': processed_content,
                        'url': page.url
                    }

                except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
                    logger.warning(f"Error with page {page_title}: {str(e)}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error with page {page_title}: {str(e)}")
                    continue

            logger.warning(f"No suitable content found for any of the search results for topic: {topic}")
            return None

        except Exception as e:
            logger.error(f"Error getting Wikipedia content: {str(e)}", exc_info=True)
            return None

    def process_content(self, content: str) -> str:
        """Process the Wikipedia content for question generation."""
        if not content:
            return ""

        try:
            # Split content into sections and take the introduction and first few relevant sections
            sections = content.split('\n\n')
            processed_sections = []
            
            for section in sections[:10]:  # Look at first 10 sections
                # Skip empty sections, references, and sections with special formatting
                if not section.strip() or \
                   section.startswith('==') or \
                   '[' in section or \
                   '|' in section or \
                   '{' in section:
                    continue
                
                # Clean the section
                cleaned_section = section.replace('\n', ' ').strip()
                if len(cleaned_section.split()) > 5:  # Only keep meaningful sections
                    processed_sections.append(cleaned_section)
                
                # Stop if we have enough content
                if len(processed_sections) >= 3:
                    break

            if not processed_sections:
                return ""

            processed_content = ' '.join(processed_sections)

            # Use NLTK for sentence tokenization if available
            if self.use_nltk:
                try:
                    sentences = nltk.sent_tokenize(processed_content)
                    # Keep only sentences that are likely to be informative
                    sentences = [s for s in sentences if len(s.split()) > 5 and len(s.split()) < 50]
                    return ' '.join(sentences)
                except Exception as e:
                    logger.warning(f"NLTK processing failed: {str(e)}")

            # Fallback to basic processing
            return processed_content

        except Exception as e:
            logger.error(f"Error processing content: {str(e)}")
            return ""

# Initialize the collector
wikipedia_collector = WikipediaCollector() 