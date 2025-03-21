"""
AutoExam Question Generator Module
--------------------------------
This module handles the generation of exam questions from text content.
It uses NLTK for text processing and OpenAI's GPT-3.5 for question enhancement.

The module provides functionality to:
- Tokenize text into sentences
- Extract relevant keywords
- Generate multiple-choice questions
- Generate fill-in-the-blank questions (basic)
"""

import os
import random
import nltk
import logging
import openai
from typing import List, Dict, Union

# Configure logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuestionGenerator:
    """
    A class to generate exam questions from text content.
    
    This class provides methods to process text and generate both
    multiple-choice and fill-in-the-blank questions using NLP
    techniques and AI enhancement.
    """

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
        """
        Perform basic sentence tokenization when NLTK is not available.
        
        Args:
            text (str): The input text to tokenize
            
        Returns:
            List[str]: A list of sentences
        """
        # Split on common sentence endings and clean the results
        text = text.replace('? ', '?|').replace('! ', '!|').replace('. ', '.|')
        sentences = text.split('|')
        return [s.strip() for s in sentences if s.strip()]

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into sentences using NLTK if available, otherwise use basic tokenization.
        
        Args:
            text (str): The input text to tokenize
            
        Returns:
            List[str]: A list of sentences
        """
        try:
            if not self.use_basic_tokenization:
                return nltk.sent_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK tokenization failed: {str(e)}")
        return self._basic_tokenize(text)

    def _extract_keywords(self, sentence: str) -> List[str]:
        """
        Extract potential keywords from a sentence for question generation.
        Uses NLTK POS tagging if available, otherwise uses length-based selection.
        
        Args:
            sentence (str): The input sentence
            
        Returns:
            List[str]: A list of potential keywords
        """
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
        """
        Use ChatGPT to create an enhanced multiple-choice question.
        
        Args:
            sentence (str): The context sentence
            keyword (str): The key concept to focus on
            
        Returns:
            Dict: A dictionary containing the question, options, answer, and explanation
                 Returns None if generation fails
        """
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("OpenAI API key not found")
                return None

            # Validate API key format
            if not (api_key.startswith('sk-') or api_key.startswith('sk-proj-')):
                logger.error("Invalid OpenAI API key format")
                return None

            # Set the API key
            openai.api_key = api_key

            # Clean and validate the sentence
            sentence = sentence.strip()
            if len(sentence.split()) < 10:
                logger.warning("Sentence too short for question generation")
                return None

            prompt = f"""Create a multiple-choice question based on this text: "{sentence}"

Focus on testing understanding of key concepts, especially around "{keyword}".

Requirements:
1. Question should be clear and test understanding (not just memorization)
2. Provide exactly 4 options labeled A, B, C, D
3. One option must be clearly correct
4. Other options should be plausible but incorrect
5. Include a brief explanation of why the correct answer is right

Format your response as a JSON object with these exact fields:
{{
    "question": "The complete question text",
    "options": ["A) first option", "B) second option", "C) third option", "D) fourth option"],
    "correct_answer": "The full text of the correct option (including the letter prefix)",
    "explanation": "A brief explanation of why this is the correct answer"
}}"""

            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a professional exam question creator."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                logger.info("Successfully received response from ChatGPT")
            except openai.error.AuthenticationError as e:
                logger.error(f"OpenAI API authentication failed: {str(e)}")
                return None
            except openai.error.RateLimitError as e:
                logger.error(f"OpenAI API rate limit exceeded: {str(e)}")
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
                import json
                try:
                    # First try json.loads for proper JSON
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # Fallback to ast.literal_eval for Python dict format
                    result = ast.literal_eval(content)

                # Validate the result
                required_fields = ['question', 'options', 'correct_answer', 'explanation']
                if not all(field in result for field in required_fields):
                    logger.error("Missing required fields in ChatGPT response")
                    return None
                if len(result['options']) != 4:
                    logger.error("Incorrect number of options in ChatGPT response")
                    return None

                return {
                    'type': 'multiple_choice',
                    'question': result['question'].strip(),
                    'options': [opt.strip() for opt in result['options']],
                    'answer': result['correct_answer'].strip(),
                    'explanation': result['explanation'].strip()
                }
            except Exception as e:
                logger.error(f"Failed to parse ChatGPT response: {str(e)}")
                logger.error(f"Raw response content: {content}")
                return None

        except Exception as e:
            logger.error(f"ChatGPT enhancement failed: {str(e)}")
            return None

    def _generate_basic_question(self, sentence: str, keyword: str) -> Dict[str, Union[str, List[str]]]:
        """
        Generate a simple fill-in-the-blank question by replacing a keyword.
        
        Args:
            sentence (str): The context sentence
            keyword (str): The word to blank out
            
        Returns:
            Dict: A dictionary containing the question, answer, and explanation
        """
        question_text = sentence.replace(keyword, "________")
        return {
            'type': 'fill_blank',
            'question': question_text,
            'answer': keyword,
            'options': [],
            'explanation': f"The word '{keyword}' fits in this context based on the original text."
        }

    def generate_questions(self, content, num_questions=5):
        """
        Generate a set of questions from the provided content.
        
        Args:
            content (Union[str, Dict]): The input text or a dictionary containing text and metadata
            num_questions (int): The number of questions to generate (default: 5)
            
        Returns:
            List[Dict]: A list of generated questions with their answers and explanations
                       Returns None if generation fails
        """
        try:
            # Handle both string and dictionary content
            if isinstance(content, dict):
                text = content.get('text', '')
                wiki_url = content.get('url', '')
                title = content.get('title', '')
            else:
                text = str(content)
                wiki_url = ''
                title = ''

            if not text.strip():
                logger.warning("Empty content provided")
                return None

            # Tokenize the text into sentences
            sentences = self._tokenize_text(text)
            if not sentences:
                logger.warning("No sentences found in content")
                return None
            
            # Filter out short sentences and those with special characters
            sentences = [s for s in sentences 
                        if len(s.split()) >= 10 and  # Minimum words
                        len(s.split()) <= 50 and  # Maximum words
                        not any(char in s for char in ['|', '{', '}', '[', ']'])]
            
            if not sentences:
                logger.warning("No valid sentences found after filtering")
                return None
            
            questions = []
            attempts = 0
            max_attempts = num_questions * 3  # Allow for some failed attempts
            
            while len(questions) < num_questions and attempts < max_attempts:
                if not sentences:
                    break
                
                # Select a random sentence
                sentence = random.choice(sentences)
                sentences.remove(sentence)  # Avoid reusing the same sentence
                
                try:
                    # Extract keywords
                    keywords = self._extract_keywords(sentence)
                    if not keywords:
                        logger.warning(f"No keywords found in sentence: {sentence[:50]}...")
                        continue
                        
                    # Select a keyword and generate a question
                    keyword = random.choice(keywords)
                    question_data = self._enhance_with_chatgpt(sentence, keyword)
                    
                    if question_data:
                        question_data['source_url'] = wiki_url
                        question_data['source_title'] = title
                        questions.append(question_data)
                        logger.info(f"Successfully generated question {len(questions)}/{num_questions}")
                
                except Exception as e:
                    logger.warning(f"Failed to generate question: {str(e)}")
                
                attempts += 1
            
            if not questions:
                logger.warning("Failed to generate any questions")
                return None
            
            return questions
            
        except Exception as e:
            logger.error(f"Error in generate_questions: {str(e)}")
            return None

# Initialize the question generator
question_generator = QuestionGenerator() 