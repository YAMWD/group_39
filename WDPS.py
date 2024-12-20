import spacy
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama
import torch
from transformers import AutoTokenizer, AutoModel
import re
from typing import Dict, List, Tuple, Optional, Union

def download_required_models():
    """Download required spaCy models if not already installed"""
    try:
        spacy.cli.download("en_core_web_trf")
    except:
        print("Model en_core_web_trf already installed")

class EntityProcessor:
    """Handles entity recognition and linking"""
    
    def __init__(self):
        # Initialize spaCy with transformer model
        self.nlp = spacy.load("en_core_web_trf")
        
        # Initialize vectorizer for entity disambiguation
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=2
        )
        
        # Define relevant entity types
        self.relevant_types = {'PERSON', 'ORG', 'GPE', 'LOC', 'FAC', 'PRODUCT', 'EVENT', 'WORK_OF_ART'}

    def process_text(self, input_text: str, output_text: str) -> Dict[str, str]:
        """Process texts to extract and link entities"""
        combined_text = f"{input_text}\n{output_text}"
        entities = self._extract_entities(combined_text)
        linked_entities = self._link_entities(entities)
        return linked_entities

    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities using spaCy"""
        doc = self.nlp(text)
        entities = []
        seen = set()
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in self.relevant_types and ent.text.lower() not in seen:
                clean_text = self._clean_entity_text(ent.text)
                if clean_text:
                    entities.append((clean_text, ent.label_))
                    seen.add(clean_text.lower())
        
        return entities

    def _clean_entity_text(self, text: str) -> str:
        """Clean and normalize entity text"""
        # Remove special characters and normalize whitespace
        cleaned = re.sub(r'[^\w\s]', ' ', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove noise words
        noise_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
        cleaned = ' '.join(word for word in cleaned.split() 
                         if len(word) > 1 and word.lower() not in noise_words)
        
        return cleaned

    def _link_entities(self, entities: List[Tuple[str, str]]) -> Dict[str, str]:
        """Link entities to Wikipedia URLs"""
        linked_entities = {}
        for entity_text, entity_type in entities:
            wiki_url = self._disambiguate_entity(entity_text, entity_type)
            if wiki_url:
                linked_entities[entity_text] = wiki_url
        return linked_entities

    def _disambiguate_entity(self, entity_text: str, entity_type: str) -> Optional[str]:
        """Disambiguate entity using VSM with type-specific disambiguation"""
        try:
            # Add type-specific search terms
            search_query = entity_text
            if entity_type == 'GPE':
                search_query = f"{entity_text} city country"
            elif entity_type == 'ORG':
                search_query = f"{entity_text} organization"
            elif entity_type == 'LOC':
                search_query = f"{entity_text} location"
            
            search_results = wikipedia.search(search_query, results=8)  # Increased results for better matching
            if not search_results:
                return None
                
            # Filter results based on entity type
            if entity_type == 'GPE':
                search_results = [r for r in search_results 
                                if not any(x in r.lower() for x in ['radio', 'sweet', 'song', 'film'])]
            elif entity_type == 'ORG':
                search_results = [r for r in search_results
                                if not any(x in r.lower() for x in ['oil', 'corporation', 'company'])]
                                
            if not search_results:
                return None

            candidates = []
            summaries = []
            
            for result in search_results:
                try:
                    summary = wikipedia.summary(result, sentences=2)
                    type_context = self._get_type_context(entity_type)
                    augmented_summary = f"{type_context} {summary}"
                    candidates.append(result)
                    summaries.append(augmented_summary)
                except:
                    continue

            if not summaries:
                return None

            entity_with_context = f"{self._get_type_context(entity_type)} {entity_text}"
            all_docs = [entity_with_context] + summaries
            
            tfidf_matrix = self.vectorizer.fit_transform(all_docs)
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            best_match_idx = similarities.argmax()
            best_match = candidates[best_match_idx]
            
            return f"https://en.wikipedia.org/wiki/{best_match.replace(' ', '_')}"
            
        except Exception as e:
            print(f"Error linking entity {entity_text}: {str(e)}")
            return None

    def _get_type_context(self, entity_type: str) -> str:
        """Get context words for entity type"""
        type_contexts = {
            'PERSON': 'person individual human',
            'ORG': 'organization company institution',
            'GPE': 'location place city country state',
            'LOC': 'location place geographic',
            'FAC': 'facility building structure',
            'PRODUCT': 'product item brand',
            'EVENT': 'event occurrence happening',
            'WORK_OF_ART': 'artwork creation piece',
            'MISC': ''
        }
        return type_contexts.get(entity_type, '')

class AnswerExtractor:
    """Handles answer extraction from LLM responses"""
    
    def __init__(self, nlp):
        self.nlp = nlp
        
        # Initialize BERT
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
        
        # Answer patterns
        self.affirmative_patterns = {
            'yes', 'correct', 'right', 'true', 'indeed', 'absolutely', 
            'definitely', 'certainly', 'surely', 'know', 'is'
        }
        self.negative_patterns = {
            'no', 'incorrect', 'wrong', 'false', 'negative', 'not'
        }

    def resolve_basic_references(self, text: str) -> str:
        """Simple rule-based coreference resolution"""
        doc = self.nlp(text)
        resolved_text = text
        recent_entities = {}
        
        # First pass: collect named entities
        for ent in doc.ents:
            if ent.label_ in {'PERSON', 'ORG', 'GPE', 'LOC', 'FAC', 'PRODUCT'}:
                recent_entities[ent.label_] = ent.text
        
        # Second pass: replace pronouns
        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "PRON":
                    replacement = None
                    
                    # Personal pronouns
                    if token.lower_ in ["he", "him", "his"]:
                        replacement = recent_entities.get("PERSON")
                    elif token.lower_ in ["she", "her", "hers"]:
                        replacement = recent_entities.get("PERSON")
                    elif token.lower_ in ["it", "its"]:
                        for label in ["ORG", "GPE", "LOC", "FAC", "PRODUCT"]:
                            if label in recent_entities:
                                replacement = recent_entities[label]
                                break
                    elif token.lower_ in ["they", "them", "their", "theirs"]:
                        for label in ["ORG", "GPE", "PERSON"]:
                            if label in recent_entities:
                                replacement = recent_entities[label]
                                break
                    
                    if replacement:
                        resolved_text = resolved_text.replace(f" {token.text} ", f" {replacement} ")
        
        return resolved_text

    def extract_answer(self, question: str, llm_response: str) -> str:
        """Extract answer based on question type"""
        resolved_response = self.resolve_basic_references(llm_response)
        question_type = self._determine_question_type(question)
        
        if question_type == 'yes_no':
            return self._extract_yes_no_answer(resolved_response)
        else:
            return "entity"

    def _determine_question_type(self, question: str) -> str:
        """Determine type of question"""
        question_lower = question.lower()
        
        if any(question_lower.startswith(pattern) for pattern in 
               ['is', 'are', 'was', 'were', 'do', 'does', 'did', 'has', 'have', 'had']):
            return 'yes_no'
            
        if '...' in question or not question.endswith('?'):
            return 'entity'
            
        return 'entity'

    def _extract_yes_no_answer(self, text: str) -> str:
        """Extract yes/no answer using multiple techniques"""
        doc = self.nlp(text.lower())
        sentiment_score = 0
        
        # First analyze the first sentence with higher weight
        first_sent = next(doc.sents)
        first_sent_text = first_sent.text.lower()
        
        # Check for immediate negation patterns
        if "surely not" in first_sent_text or "certainly not" in first_sent_text:
            return 'no'
            
        # Check for strong affirmative indicators followed by negation
        strong_indicators = ['surely', 'certainly', 'definitely', 'absolutely']
        for indicator in strong_indicators:
            if indicator in first_sent_text:
                # Check if followed by negation
                words = first_sent_text.split()
                try:
                    idx = words.index(indicator)
                    if idx + 1 < len(words) and words[idx + 1] in ['not', "n't"]:
                        return 'no'
                except ValueError:
                    continue
        
        # Check for statement structure "X is Y" in first sentence
        subject_found = False
        is_found = False
        for token in first_sent:
            if token.dep_ == 'nsubj':
                subject_found = True
            if token.lemma_ == 'be' and subject_found:
                is_found = True
                if not any(child.dep_ == 'neg' for child in token.children):
                    sentiment_score += 1
        
        # Continue with regular pattern matching
        weight = 1.0
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            # Check patterns
            if any(word in sent_text for word in self.affirmative_patterns):
                sentiment_score += 1 * weight
            if any(word in sent_text for word in self.negative_patterns):
                sentiment_score -= 1 * weight
            
            # Check negation using dependency parsing
            for token in sent:
                if token.dep_ == 'neg':
                    sentiment_score -= 0.5 * weight
                    
                if token.pos_ == "VERB":
                    has_negation = any(child.dep_ == 'neg' for child in token.children)
                    if has_negation:
                        sentiment_score -= 0.3 * weight
            
            weight *= 0.7
        
        if sentiment_score > 0:
            return 'yes'
        elif sentiment_score < 0:
            return 'no'
        return 'unknown'

class LlamaProcessor:
    """Handles Llama model interactions"""
    
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=8,
            n_batch=512
        )

    def get_response(self, question: str) -> str:
        """Get response from Llama model"""
        try:
            response = self.llm(
                question,
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                stop=["</s>"],
                echo=False
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            print(f"Error getting LLM response: {str(e)}")
            return ""

class QuestionProcessor:
    """Main class for processing questions"""
    
    def __init__(self, model_path: str):
        self.llm_processor = LlamaProcessor(model_path)
        self.entity_processor = EntityProcessor()
        self.answer_extractor = AnswerExtractor(self.entity_processor.nlp)

    def process_question(self, question_id: str, question: str) -> List[str]:
        """Process a single question"""
        try:
            # Get LLM response
            llm_response = self.llm_processor.get_response(question)
            
            # Process entities
            entities = self.entity_processor.process_text(question, llm_response)
            
            # Extract answer
            answer = self.answer_extractor.extract_answer(question, llm_response)
            
            # Format output
            output_lines = [
                f'{question_id}\tR"{llm_response}"',
                f'{question_id}\tA"{answer}"'
            ]
            
            for entity, url in entities.items():
                output_lines.append(f'{question_id}\tE"{entity}"\t"{url}"')
            
            return output_lines
            
        except Exception as e:
            print(f"Error processing question {question_id}: {str(e)}")
            return [f'{question_id}\tERROR"{str(e)}"']

def main():
    """Main function"""
    # Download required models
    download_required_models()
    
    # Initialize processor
    model_path = "C:/Users/yaucy/OneDrive/Documents/University Work/VU/Web Data Processing Systems/Assignment/wdps/models/llama-2-7b.Q4_K_M.gguf"
    processor = QuestionProcessor(model_path)
    
    try:
        with open('input.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        with open('output.txt', 'w', encoding='utf-8') as f:
            for line in lines:
                question_id, question = line.strip().split('\t')
                output_lines = processor.process_question(question_id, question)
                f.write('\n'.join(output_lines) + '\n')
                
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        with open('error_log.txt', 'w', encoding='utf-8') as f:
            f.write(f"Fatal error occurred: {str(e)}\n")

if __name__ == "__main__":
    main()