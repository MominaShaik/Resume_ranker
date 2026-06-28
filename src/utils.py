"""
Utility functions for text preprocessing, embedding generation, and formatting.
Optimized for CPU-only execution with minimal memory footprint.
"""

import re
from typing import List, Dict, Optional, Set
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
import math


class TextPreprocessor:
    """Lightweight text preprocessing for candidate and job description data."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text for embedding.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\.,;:\-]', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_skills_text(skills: List[str]) -> str:
        """Convert skills list to searchable text."""
        if not skills:
            return ""
        return " ".join([skill.strip() for skill in skills if skill])
    
    @staticmethod
    def extract_experience_text(experience: List[Dict]) -> str:
        """
        Extract relevant text from experience entries.
        
        Args:
            experience: List of experience dictionaries
            
        Returns:
            Combined text string
        """
        if not experience:
            return ""
        
        texts = []
        for exp in experience:
            parts = []
            if 'title' in exp:
                parts.append(exp['title'])
            if 'company' in exp:
                parts.append(exp['company'])
            if 'description' in exp:
                parts.append(exp['description'])
            texts.append(" ".join(parts))
        
        return " ".join(texts)
    
    @staticmethod
    def extract_projects_text(projects: List[Dict]) -> str:
        """Extract relevant text from project entries."""
        if not projects:
            return ""
        
        texts = []
        for proj in projects:
            parts = []
            if 'name' in proj:
                parts.append(proj['name'])
            if 'description' in proj:
                parts.append(proj['description'])
            if 'technologies' in proj:
                parts.append(" ".join(proj['technologies']))
            texts.append(" ".join(parts))
        
        return " ".join(texts)
    
    @staticmethod
    def build_candidate_text(candidate: Dict) -> str:
        """
        Build a comprehensive text representation of a candidate for embedding.
        
        Args:
            candidate: Candidate dictionary
            
        Returns:
            Combined text string
        """
        parts = []
        
        # Skills
        skills = TextPreprocessor.extract_skills_text(candidate.get('skills', []))
        if skills:
            parts.append(f"Skills: {skills}")
        
        # Experience
        exp_text = TextPreprocessor.extract_experience_text(candidate.get('experience', []))
        if exp_text:
            parts.append(f"Experience: {exp_text}")
        
        # Projects
        proj_text = TextPreprocessor.extract_projects_text(candidate.get('projects', []))
        if proj_text:
            parts.append(f"Projects: {proj_text}")
        
        # Education
        education = candidate.get('education', [])
        if education:
            edu_texts = []
            for edu in education:
                if 'degree' in edu:
                    edu_texts.append(edu['degree'])
                if 'field' in edu:
                    edu_texts.append(edu['field'])
            if edu_texts:
                parts.append(f"Education: {' '.join(edu_texts)}")
        
        combined = " ".join(parts)
        return TextPreprocessor.clean_text(combined)


class EmbeddingGenerator:
    """
    CPU-optimized embedding generator using sentence-transformers.
    Uses lightweight models for fast inference on CPU.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model
                       Recommended: 'all-MiniLM-L6-v2' (fast, 384-dim)
                                    'bge-small-en-v1.5' (better quality, 384-dim)
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("Model loaded successfully")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            
        Returns:
            numpy array of embeddings (shape: [len(texts), embedding_dim])
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        return self.encode([text])[0]
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def compute_batch_similarity(self, query_embedding: np.ndarray, 
                                 candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity between query and multiple candidates.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Array of candidate embeddings
            
        Returns:
            Array of similarity scores
        """
        if len(candidate_embeddings) == 0:
            return np.array([])
        
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        candidate_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute dot products
        similarities = np.dot(candidate_norms, query_norm)
        return similarities


class BehavioralSignalScorer:
    """
    Scorer for the 23 redrob behavioral signals.
    Implements boost factors and penalty logic for candidate quality assessment.
    """
    
    # Signal weights (positive = boost, negative = penalty)
    SIGNAL_WEIGHTS = {
        # Positive signals (boosts)
        'profile_completion': 0.1,
        'verified_skills': 0.15,
        'response_rate': 0.12,
        'last_active_days': -0.05,  # More recent = higher score
        'application_count': 0.08,
        'interview_attendance': 0.15,
        'offer_acceptance_rate': 0.1,
        'platform_engagement': 0.1,
        'referral_count': 0.08,
        'skill_endorsements': 0.1,
        
        # Negative signals (penalties)
        'profile_mismatch': -0.5,
        'failed_verification': -0.8,
        'suspicious_activity': -0.7,
        'job_hopping_frequency': -0.5,  # Increased penalty for title-chasers
        'employment_gaps': -0.15,
        'location_mismatch': -0.2,
        'salary_expectation_mismatch': -0.1,
        'communication_issues': -0.2,
        'negative_feedback': -0.4,
        'incomplete_profile': -0.1,
        'duplicate_profile': -0.9,
        'bot_like_behavior': -0.95,
        
        # Redrob-specific negative signals
        'consulting_only_background': -0.6,  # Only worked at consulting firms
        'framework_enthusiast': -0.4,  # LangChain tutorials without systems thinking
        'cv_speech_robotics_primary': -0.5,  # Primary expertise not NLP/IR
        'closed_source_only': -0.4,  # No external validation
        'pure_research_background': -0.7,  # No production deployment
        'recent_llm_only': -0.5,  # Only recent LangChain/OpenAI experience
        'no_recent_code': -0.6,  # No production code in 18+ months
    }
    
    @staticmethod
    def score_signals(signals: Dict) -> float:
        """
        Compute composite behavioral signal score.
        
        Args:
            signals: Dictionary of redrob_signals
            
        Returns:
            Composite score (can be negative for bad candidates)
        """
        if not signals:
            return 0.0
        
        total_score = 0.0
        
        for signal_name, value in signals.items():
            if signal_name in BehavioralSignalScorer.SIGNAL_WEIGHTS:
                weight = BehavioralSignalScorer.SIGNAL_WEIGHTS[signal_name]
                
                # Normalize value to 0-1 range if needed
                normalized_value = BehavioralSignalScorer._normalize_signal(signal_name, value)
                total_score += weight * normalized_value
        
        return total_score
    
    @staticmethod
    def compute_multiplier(signals: Dict) -> float:
        """
        Compute dynamic multiplier for high-intent candidates.
        Applied to base semantic score to boost high-quality candidates.
        
        Args:
            signals: Dictionary of redrob_signals
            
        Returns:
            Multiplier (default 1.0, higher for high-intent candidates)
        """
        if not signals:
            return 1.0
        
        multiplier = 1.0
        
        # Boost for high response rate
        response_rate = BehavioralSignalScorer._normalize_signal('response_rate', signals.get('response_rate', 0))
        if response_rate > 0.8:
            multiplier += 0.1
        elif response_rate > 0.6:
            multiplier += 0.05
        
        # Boost for verified skills
        if signals.get('verified_skills', False):
            multiplier += 0.08
        
        # Boost for high profile completion
        profile_completion = BehavioralSignalScorer._normalize_signal('profile_completion', signals.get('profile_completion', 0))
        if profile_completion > 0.9:
            multiplier += 0.05
        elif profile_completion > 0.8:
            multiplier += 0.03
        
        # Boost for recent activity
        last_active = BehavioralSignalScorer._normalize_signal('last_active_days', signals.get('last_active_days', 365))
        if last_active > 0.8:  # Very recent
            multiplier += 0.05
        
        # Boost for interview attendance
        interview_attendance = BehavioralSignalScorer._normalize_signal('interview_attendance', signals.get('interview_attendance', 0))
        if interview_attendance > 0.9:
            multiplier += 0.05
        
        # Cap multiplier at reasonable maximum
        return min(multiplier, 1.3)
    
    @staticmethod
    def compute_hard_penalty(signals: Dict) -> float:
        """
        Compute hard penalty for trap candidates.
        Applied as a subtraction from final score.
        
        Args:
            signals: Dictionary of redrob_signals
            
        Returns:
            Penalty score (0 to 1, higher = worse)
        """
        if not signals:
            return 0.0
        
        penalty = 0.0
        
        # Heavy penalties for trap signals
        if signals.get('job_hopping_frequency', 0) > 3:
            penalty += 0.4
        
        if signals.get('failed_verification', False):
            penalty += 0.8
        
        if signals.get('suspicious_activity', False):
            penalty += 0.7
        
        if signals.get('duplicate_profile', False):
            penalty += 0.9
        
        if signals.get('bot_like_behavior', False):
            penalty += 0.95
        
        if signals.get('framework_enthusiast', False):
            penalty += 0.3
        
        if signals.get('recent_llm_only', False):
            penalty += 0.4
        
        if signals.get('no_recent_code', False):
            penalty += 0.5
        
        return min(penalty, 1.0)
    
    @staticmethod
    def _normalize_signal(signal_name: str, value: any) -> float:
        """
        Normalize signal value to 0-1 range.
        
        Args:
            signal_name: Name of the signal
            value: Raw signal value
            
        Returns:
            Normalized value between 0 and 1
        """
        # Handle different value types
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        
        if isinstance(value, (int, float)):
            # For percentage-based signals
            if signal_name in ['response_rate', 'offer_acceptance_rate', 'profile_completion']:
                return min(max(value / 100.0, 0.0), 1.0)
            
            # For count-based signals (cap at reasonable max)
            if signal_name in ['application_count', 'referral_count', 'skill endorsements']:
                return min(value / 10.0, 1.0)
            
            # For days-based (inverse - fewer days is better)
            if signal_name == 'last_active_days':
                return max(1.0 - (value / 365.0), 0.0)
            
            # For frequency-based
            if signal_name == 'job_hopping_frequency':
                return min(value / 5.0, 1.0)
            
            return min(max(float(value), 0.0), 1.0)
        
        if isinstance(value, str):
            if value.lower() in ['true', 'yes', 'high']:
                return 1.0
            elif value.lower() in ['false', 'no', 'low']:
                return 0.0
        
        return 0.0
    
    @staticmethod
    def has_hard_penalty(signals: Dict) -> bool:
        """
        Check if candidate has hard penalty signals that should disqualify them.
        
        Args:
            signals: Dictionary of redrob_signals
            
        Returns:
            True if candidate should be disqualified
        """
        hard_penalties = [
            'failed_verification',
            'suspicious_activity',
            'duplicate_profile',
            'bot_like_behavior',
            'pure_research_background',  # Redrob-specific: no production deployment
        ]
        
        for penalty in hard_penalties:
            if signals.get(penalty, False):
                return True
        
        return False


class ExperienceMatcher:
    """Matcher for evaluating experience alignment with job requirements."""
    
    @staticmethod
    def calculate_experience_score(candidate: Dict, jd_requirements: Dict) -> float:
        """
        Calculate experience match score based on years and relevance.
        
        Args:
            candidate: Candidate dictionary
            jd_requirements: Parsed job description requirements
            
        Returns:
            Experience score (0 to 1)
        """
        experience = candidate.get('experience', [])
        if not experience:
            return 0.0
        
        # Calculate total years of experience
        total_years = ExperienceMatcher._calculate_total_years(experience)
        
        # Check against minimum required
        min_required = jd_requirements.get('min_experience_years', 0)
        
        if total_years < min_required:
            # Penalize for not meeting minimum
            return max(0.0, total_years / min_required * 0.5)
        
        # Cap score at 1.0 (more years doesn't always mean better)
        return min(1.0, total_years / (min_required * 2) if min_required > 0 else 0.5)
    
    @staticmethod
    def _calculate_total_years(experience: List[Dict]) -> float:
        """Calculate total years from experience entries."""
        total = 0.0
        for exp in experience:
            if 'years' in exp:
                total += float(exp['years'])
            elif 'duration' in exp:
                # Try to parse duration string
                duration_str = str(exp['duration'])
                if 'year' in duration_str.lower():
                    import re
                    match = re.search(r'(\d+)', duration_str)
                    if match:
                        total += float(match.group(1))
        return total
    
    @staticmethod
    def passes_hard_filter(candidate: Dict, jd_requirements: Dict) -> bool:
        """
        Apply hard filtering based on non-negotiable requirements.
        This is Stage 1 of the cascading funnel.
        
        Args:
            candidate: Candidate dictionary
            jd_requirements: Parsed job description requirements
            
        Returns:
            True if candidate passes hard filters
        """
        # Check minimum experience (with some tolerance)
        min_required = jd_requirements.get('min_experience_years', 0)
        if min_required > 0:
            total_years = ExperienceMatcher._calculate_total_years(
                candidate.get('experience', [])
            )
            # Allow 2-year tolerance below minimum
            if total_years < max(0, min_required - 2):
                return False
        
        # Check for hard penalty signals
        signals = candidate.get('redrob_signals', {})
        if BehavioralSignalScorer.has_hard_penalty(signals):
            return False
        
        # Check for Redrob-specific disqualifiers
        if signals.get('consulting_only_background', False):
            return False
        if signals.get('pure_research_background', False):
            return False
        
        return True


class LexicalFilter:
    """
    Ultra-fast lexical/BM25 filter for Stage 1 pruning.
    Reduces 100K candidates to ~2K before expensive semantic operations.
    """
    
    def __init__(self, jd_requirements: Dict):
        """
        Initialize the lexical filter with JD requirements.
        
        Args:
            jd_requirements: Parsed job description requirements
        """
        self.jd_requirements = jd_requirements
        self.required_skills = set(jd_requirements.get('required_skills', []))
        
        # Extract key terms from JD for matching
        jd_text = jd_requirements.get('full_text', '').lower()
        self.key_terms = self._extract_key_terms(jd_text)
        
        # Build document frequency for BM25 (will be updated during filtering)
        self.doc_freq = Counter()
        self.total_docs = 0
    
    def _extract_key_terms(self, text: str) -> Set[str]:
        """Extract important technical terms from JD text."""
        # Common technical terms to look for
        tech_terms = {
            'python', 'java', 'c++', 'javascript', 'typescript',
            'tensorflow', 'pytorch', 'keras', 'scikit', 'pandas',
            'nlp', 'computer vision', 'machine learning', 'deep learning',
            'embeddings', 'retrieval', 'ranking', 'vector', 'search',
            'recommendation', 'llm', 'gpt', 'transformer', 'bert',
            'docker', 'kubernetes', 'aws', 'gcp', 'azure',
            'sql', 'nosql', 'mongodb', 'postgresql',
            'redis', 'elasticsearch', 'pinecone', 'faiss', 'weaviate',
            'mlops', 'deployment', 'production', 'api'
        }
        
        found_terms = set()
        text_lower = text.lower()
        for term in tech_terms:
            if term in text_lower:
                found_terms.add(term)
        
        return found_terms
    
    def compute_bm25_score(self, candidate: Dict, avg_doc_length: float = 50) -> float:
        """
        Compute BM25 score for a candidate.
        
        Args:
            candidate: Candidate dictionary
            avg_doc_length: Average document length for normalization
            
        Returns:
            BM25 score (higher is better)
        """
        # Build candidate text
        candidate_text = TextPreprocessor.build_candidate_text(candidate).lower()
        tokens = candidate_text.split()
        doc_length = len(tokens)
        
        if doc_length == 0:
            return 0.0
        
        # Count term frequencies
        term_freq = Counter(tokens)
        
        # BM25 parameters
        k1 = 1.5  # Term saturation parameter
        b = 0.75  # Length normalization parameter
        
        score = 0.0
        
        for term in self.key_terms:
            if term in term_freq:
                tf = term_freq[term]
                df = self.doc_freq.get(term, 1)
                idf = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
                
                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                score += idf * (numerator / denominator)
        
        return score
    
    def compute_keyword_overlap_score(self, candidate: Dict) -> float:
        """
        Compute simple keyword overlap score.
        
        Args:
            candidate: Candidate dictionary
            
        Returns:
            Keyword overlap score (0 to 1)
        """
        candidate_skills = set(candidate.get('skills', []))
        candidate_text = TextPreprocessor.build_candidate_text(candidate).lower()
        
        # Check skill overlap
        skill_overlap = len(self.required_skills & candidate_skills)
        
        # Check key term overlap in text
        term_overlap = sum(1 for term in self.key_terms if term in candidate_text)
        
        # Normalize
        max_possible = len(self.required_skills) + len(self.key_terms)
        if max_possible == 0:
            return 0.0
        
        return (skill_overlap * 2 + term_overlap) / max_possible
    
    def update_doc_freq(self, candidate: Dict):
        """Update document frequency statistics for BM25."""
        candidate_text = TextPreprocessor.build_candidate_text(candidate).lower()
        tokens = set(candidate_text.split())
        
        for term in self.key_terms:
            if term in tokens:
                self.doc_freq[term] += 1
        
        self.total_docs += 1
    
    def should_keep_candidate(self, candidate: Dict, threshold: float = 0.1) -> bool:
        """
        Fast heuristic to decide if candidate should pass Stage 1 filter.
        
        Args:
            candidate: Candidate dictionary
            threshold: Minimum score threshold
            
        Returns:
            True if candidate should be kept for Stage 2
        """
        # Check for product company experience (fast filter)
        experience = candidate.get('experience', [])
        has_product_exp = any(
            'consulting' not in exp.get('company', '').lower() and 
            'tcs' not in exp.get('company', '').lower() and
            'infosys' not in exp.get('company', '').lower() and
            'wipro' not in exp.get('company', '').lower()
            for exp in experience
        )
        
        # Check for ranking/search/retrieval experience
        has_ranking_exp = False
        for exp in experience:
            exp_text = (exp.get('title', '') + ' ' + exp.get('description', '')).lower()
            if any(term in exp_text for term in ['ranking', 'retrieval', 'recommendation', 'search', 'match']):
                has_ranking_exp = True
                break
        
        # Compute keyword overlap
        keyword_score = self.compute_keyword_overlap_score(candidate)
        
        # Must pass at least one criterion
        passes_filter = (
            has_product_exp or 
            has_ranking_exp or 
            keyword_score >= threshold
        )
        
        return passes_filter
