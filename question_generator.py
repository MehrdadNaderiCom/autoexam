import os
import random
import nltk
import logging
import openai
from typing import List, Dict, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

class QuestionGenerator:
    def __init__(self):
        """Initialize the QuestionGenerator."""
        self.use_basic_tokenization = True
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/averaged_perceptron_tagger')
            self.use_basic_tokenization = False
            logger.info("NLTK data found, using advanced tokenization")
        except LookupError:
            logger.info("NLTK data not found, using basic tokenization")

    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic sentence tokenization."""
        # Split on common sentence endings
        text = text.replace('? ', '?|').replace('! ', '!|').replace('. ', '.|')
        sentences = text.split('|')
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into sentences using either NLTK or basic tokenization."""
        try:
            if not self.use_basic_tokenization:
                return nltk.sent_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {str(e)}")
        return self._basic_tokenize(text)

    def _extract_keywords(self, sentence: str) -> List[str]:
        """Extract potential keywords from a sentence."""
        try:
            if not self.use_basic_tokenization:
                words = nltk.word_tokenize(sentence)
                pos_tags = nltk.pos_tag(words)
                return [word for word, pos in pos_tags if pos.startswith(('NN', 'VB', 'JJ'))]
        except Exception as e:
            logger.warning(f"NLTK keyword extraction failed: {str(e)}")
        
        # Fallback: return words with 4 or more characters
        words = sentence.split()
        return [word for word in words if len(word) >= 4 and word.isalnum()]

    def _enhance_with_chatgpt(self, sentence: str, keyword: str) -> Dict[str, Union[str, List[str]]]:
        """Use ChatGPT to create a multiple-choice question."""
        try:
            if not openai.api_key:
                logger.warning("OpenAI API key not found")
                return None

            # Validate API key format
            if not (openai.api_key.startswith('sk-') or openai.api_key.startswith('sk-proj-')):
                logger.error("Invalid OpenAI API key format")
                return None

            prompt = f"""Create a high-quality multiple-choice question based on this text: "{sentence}"

Focus on testing understanding of key concepts, especially around "{keyword}".

Requirements:
1. Question should be clear and professional
2. Provide exactly 4 options (A, B, C, D)
3. One option must be clearly correct
4. Other options should be plausible but incorrect
5. Include a brief explanation of why the correct answer is right

Format your response as a JSON object with these exact fields:
{{
    "question": "The complete question text",
    "options": ["option A", "option B", "option C", "option D"],
    "correct_answer": "The full text of the correct option",
    "explanation": "A brief explanation of why this is the correct answer"
}}"""

            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=500
                )
                logger.info("Successfully received response from ChatGPT")
            except openai.error.AuthenticationError as e:
                logger.error(f"OpenAI API authentication failed: {str(e)}")
                return None
            except openai.error.APIError as e:
                logger.error(f"OpenAI API error: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error calling OpenAI API: {str(e)}")
                return None

            # Parse the response
            content = response.choices[0].message.content
            try:
                # Try to evaluate the JSON-like string (safer than eval)
                import ast
                result = ast.literal_eval(content)
                return {
                    'type': 'multiple_choice',
                    'question': result['question'],
                    'options': result['options'],
                    'answer': result['correct_answer'],
                    'explanation': result['explanation']
                }
            except Exception as e:
                logger.error(f"Failed to parse ChatGPT response: {str(e)}")
                logger.error(f"Raw response content: {content}")
                return None

        except Exception as e:
            logger.error(f"ChatGPT enhancement failed: {str(e)}")
            return None

    def _generate_basic_question(self, sentence: str, keyword: str) -> Dict[str, Union[str, List[str]]]:
        """Generate a basic fill-in-the-blank question."""
        question_text = sentence.replace(keyword, "________")
        return {
            'type': 'fill_blank',
            'question': question_text,
            'answer': keyword,
            'options': [],
            'explanation': f"The word '{keyword}' fits in this context based on the original text."
        }

    def generate_questions(self, content, num_questions=5):
        """Generate multiple-choice questions from the given content."""
        try:
            # Handle both string and dictionary content
            if isinstance(content, dict):
                text = content.get('text', '')
                wiki_url = content.get('url', '')
            else:
                text = str(content)
                wiki_url = ''

            sentences = self._tokenize_text(text)
            if not sentences:
                return []
            
            questions = []
            attempts = 0
            max_attempts = num_questions * 3  # Allow for some failed attempts
            
            while len(questions) < num_questions and attempts < max_attempts:
                if not sentences:
                    break
                
                sentence = random.choice(sentences)
                sentences.remove(sentence)  # Avoid reusing the same sentence
                
                try:
                    keywords = self._extract_keywords(sentence)
                    if not keywords:
                        continue
                        
                    keyword = random.choice(keywords)
                    
                    # Generate multiple-choice question
                    question_data = self._enhance_with_chatgpt(sentence, keyword)
                    
                    if question_data:
                        question_data['source_url'] = wiki_url
                        questions.append(question_data)
                
                except Exception as e:
                    logger.warning(f"Failed to generate question: {str(e)}")
                
                attempts += 1
            
            return questions if questions else None
            
        except Exception as e:
            logger.error(f"Error in generate_questions: {str(e)}")
            return None

# Initialize the question generator
question_generator = QuestionGenerator() 