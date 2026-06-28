"""
Core ranking module implementing hybrid scoring mechanism.
Combines semantic similarity, experience matching, and behavioral signals.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.utils import (
    TextPreprocessor,
    EmbeddingGenerator,
    BehavioralSignalScorer,
    ExperienceMatcher,
    LexicalFilter
)


@dataclass
class RankedCandidate:
    """Data structure for a ranked candidate with scores."""
    candidate_id: str
    semantic_score: float
    experience_score: float
    behavioral_score: float
    penalty_score: float
    multiplier: float
    final_score: float
    reasoning: str


class MultistageRanker:
    """
    Multistage cascading funnel ranking system for 100K candidate processing.
    
    Architecture:
    Stage 1: Ultra-fast lexical/BM25 filter (100K → ~2K)
    Stage 2: Semantic embedding matching (~2K → ~500)
    Stage 3: Behavioral signal scoring with multipliers (~500 → Top 100)
    
    Final Score = (Semantic × Experience × Behavioral × Multiplier) - Hard Penalty
    """
    
    def __init__(
        self,
        semantic_weight: float = 0.35,
        experience_weight: float = 0.25,
        behavioral_weight: float = 0.25,
        embedding_model: str = 'all-MiniLM-L6-v2',
        stage1_target: int = 2000,
        stage2_target: int = 500
    ):
        """
        Initialize the multistage ranker.
        
        Args:
            semantic_weight: Weight for semantic similarity score
            experience_weight: Weight for experience matching score
            behavioral_weight: Weight for behavioral signals score
            embedding_model: Name of sentence-transformers model to use
            stage1_target: Target number of candidates after Stage 1 filtering
            stage2_target: Target number of candidates after Stage 2 filtering
        """
        self.semantic_weight = semantic_weight
        self.experience_weight = experience_weight
        self.behavioral_weight = behavioral_weight
        self.stage1_target = stage1_target
        self.stage2_target = stage2_target
        
        # Initialize components
        self.embedding_generator = None
        self.lexical_filter = None
        
        # Cache for JD embedding
        self.jd_embedding = None
        self.jd_text = None
        self.jd_requirements = None
    
    def set_job_description(self, jd_requirements: Dict):
        """
        Set the job description and initialize all components.
        
        Args:
            jd_requirements: Parsed job description dictionary
        """
        self.jd_requirements = jd_requirements
        
        # Initialize lexical filter for Stage 1
        self.lexical_filter = LexicalFilter(jd_requirements)
        
        # Initialize embedding generator for Stage 2
        self.embedding_generator = EmbeddingGenerator('all-MiniLM-L6-v2')
        
        # Build JD text for embedding
        jd_parts = []
        
        if jd_requirements.get('title'):
            jd_parts.append(f"Role: {jd_requirements['title']}")
        
        if jd_requirements.get('required_skills'):
            jd_parts.append(f"Required Skills: {' '.join(jd_requirements['required_skills'])}")
        
        if jd_requirements.get('responsibilities'):
            jd_parts.append(f"Responsibilities: {' '.join(jd_requirements['responsibilities'])}")
        
        if jd_requirements.get('requirements'):
            jd_parts.append(f"Requirements: {' '.join(jd_requirements['requirements'])}")
        
        self.jd_text = " ".join(jd_parts)
        self.jd_text = TextPreprocessor.clean_text(self.jd_text)
        
        # Pre-compute JD embedding
        self.jd_embedding = self.embedding_generator.encode_single(self.jd_text)
    
    def stage1_lexical_filter(self, candidate_generator, target_count: int = None) -> List[Dict]:
        """
        Stage 1: Ultra-fast lexical/BM25 filter.
        Reduces 100K candidates to ~2K using streaming and heuristics.
        
        Args:
            candidate_generator: Generator yielding candidate dictionaries
            target_count: Target number of candidates to retain (default: self.stage1_target)
            
        Returns:
            Filtered list of candidate dictionaries
        """
        if target_count is None:
            target_count = self.stage1_target
        
        candidates_with_scores = []
        
        for candidate in candidate_generator:
            # Convert Candidate object to dict if needed
            if hasattr(candidate, 'to_dict'):
                candidate_dict = candidate.to_dict()
            else:
                candidate_dict = candidate
            
            # Apply hard filter (non-negotiable requirements)
            if not ExperienceMatcher.passes_hard_filter(candidate_dict, self.jd_requirements):
                continue
            
            # Update document frequency for BM25
            self.lexical_filter.update_doc_freq(candidate_dict)
            
            # Check if candidate should pass Stage 1
            if self.lexical_filter.should_keep_candidate(candidate_dict):
                # Compute BM25 score for ranking within Stage 1
                bm25_score = self.lexical_filter.compute_bm25_score(candidate_dict)
                candidates_with_scores.append((candidate_dict, bm25_score))
        
        # Sort by BM25 score and keep top target_count
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
        filtered = [c[0] for c in candidates_with_scores[:target_count]]
        
        return filtered
    
    def stage2_semantic_match(self, candidates: List[Dict], target_count: int = None) -> List[Dict]:
        """
        Stage 2: Semantic embedding matching.
        Computes vector similarities on filtered pool (~2K → ~500).
        
        Args:
            candidates: List of candidate dictionaries from Stage 1
            target_count: Target number of candidates to retain (default: self.stage2_target)
            
        Returns:
            Filtered list of candidate dictionaries sorted by semantic score
        """
        if target_count is None:
            target_count = self.stage2_target
        
        # Build candidate texts
        candidate_texts = []
        for candidate in candidates:
            text = TextPreprocessor.build_candidate_text(candidate)
            candidate_texts.append(text)
        
        # Batch encode candidates
        candidate_embeddings = self.embedding_generator.encode(candidate_texts, batch_size=32)
        
        # Compute similarities
        similarities = self.embedding_generator.compute_batch_similarity(
            self.jd_embedding, candidate_embeddings
        )
        
        # Pair candidates with scores
        candidates_with_scores = list(zip(candidates, similarities))
        
        # Sort by semantic similarity and keep top target_count
        candidates_with_scores.sort(key=lambda x: x[1], reverse=True)
        filtered = [c[0] for c in candidates_with_scores[:target_count]]
        
        return filtered
    
    def stage3_behavioral_scoring(self, candidates: List[Dict], top_k: int = 100) -> List[RankedCandidate]:
        """
        Stage 3: Behavioral signal scoring with multipliers.
        Applies dynamic boosts and hard penalties to produce final ranking.
        
        Args:
            candidates: List of candidate dictionaries from Stage 2
            top_k: Number of top candidates to output
            
        Returns:
            List of RankedCandidate objects sorted by final_score
        """
        ranked = []
        
        for candidate in candidates:
            try:
                # Convert to dict if needed
                if hasattr(candidate, 'to_dict'):
                    candidate_dict = candidate.to_dict()
                else:
                    candidate_dict = candidate
                
                # 1. Semantic Score (already computed in Stage 2, recompute for precision)
                semantic_score = self._compute_semantic_score(candidate_dict)
                
                # 2. Experience Score
                experience_score = ExperienceMatcher.calculate_experience_score(
                    candidate_dict, self.jd_requirements
                )
                
                # 3. Behavioral Score
                signals = candidate_dict.get('redrob_signals', {})
                behavioral_score = BehavioralSignalScorer.score_signals(signals)
                
                # 4. Dynamic Multiplier for high-intent candidates
                multiplier = BehavioralSignalScorer.compute_multiplier(signals)
                
                # 5. Hard Penalty for trap candidates
                penalty_score = BehavioralSignalScorer.compute_hard_penalty(signals)
                
                # 6. Final Composite Score with multiplier
                # Formula: (Semantic × Experience × Behavioral × Multiplier) - Hard Penalty
                base_score = (
                    self.semantic_weight * semantic_score +
                    self.experience_weight * experience_score +
                    self.behavioral_weight * behavioral_score
                )
                
                final_score = (base_score * multiplier) - penalty_score
                
                # 7. Generate reasoning
                reasoning = self._generate_reasoning(
                    candidate_dict, semantic_score, experience_score, behavioral_score, multiplier
                )
                
                ranked_candidate = RankedCandidate(
                    candidate_id=candidate_dict.get('candidate_id', ''),
                    semantic_score=semantic_score,
                    experience_score=experience_score,
                    behavioral_score=behavioral_score,
                    penalty_score=penalty_score,
                    multiplier=multiplier,
                    final_score=final_score,
                    reasoning=reasoning
                )
                ranked.append(ranked_candidate)
                
            except Exception as e:
                print(f"Error ranking candidate: {e}")
                continue
        
        # Sort by final score (descending)
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        
        return ranked[:top_k]
    
    def _compute_semantic_score(self, candidate: Dict) -> float:
        """
        Compute semantic similarity between candidate and job description.
        
        Args:
            candidate: Candidate dictionary
            
        Returns:
            Semantic similarity score (0 to 1)
        """
        candidate_text = TextPreprocessor.build_candidate_text(candidate)
        
        if not candidate_text:
            return 0.0
        
        candidate_embedding = self.embedding_generator.encode_single(candidate_text)
        similarity = self.embedding_generator.compute_similarity(
            self.jd_embedding, candidate_embedding
        )
        
        return float(similarity)
    
    def _generate_reasoning(
        self,
        candidate: Dict,
        semantic_score: float,
        experience_score: float,
        behavioral_score: float,
        multiplier: float
    ) -> str:
        """
        Generate 1-2 sentence reasoning for the candidate ranking.
        
        Args:
            candidate: Candidate dictionary
            semantic_score: Semantic similarity score
            experience_score: Experience match score
            behavioral_score: Behavioral signal score
            multiplier: Dynamic multiplier applied
            
        Returns:
            Human-readable reasoning string (1-2 sentences)
        """
        parts = []
        
        skills = candidate.get('skills', [])[:3]
        experience = candidate.get('experience', [])
        
        # Check for production ML/embeddings experience
        has_embeddings_exp = any(
            'embedding' in s.lower() or 'vector' in s.lower() or 'retrieval' in s.lower()
            for s in candidate.get('skills', [])
        )
        
        # Check for ranking/search/recommendation experience
        has_ranking_exp = False
        for exp in experience:
            exp_text = (exp.get('title', '') + ' ' + exp.get('description', '')).lower()
            if any(term in exp_text for term in ['ranking', 'retrieval', 'recommendation', 'search', 'match']):
                has_ranking_exp = True
                break
        
        # Check for product company experience
        has_product_exp = any(
            'consulting' not in exp.get('company', '').lower() and 
            'tcs' not in exp.get('company', '').lower() and
            'infosys' not in exp.get('company', '').lower() and
            'wipro' not in exp.get('company', '').lower()
            for exp in experience
        )
        
        # Build reasoning based on Redrob-specific criteria
        if has_ranking_exp and has_product_exp:
            if experience:
                top_company = experience[0].get('company', 'a product company')
                parts.append(f"Shipped ranking/retrieval systems at {top_company}")
        
        if has_embeddings_exp:
            parts.append("Production experience with embeddings-based retrieval")
        
        if semantic_score > 0.65:
            if skills:
                parts.append(f"Strong alignment with {', '.join(skills[:2])} requirements")
        
        if experience_score > 0.6 and experience:
            years = sum(exp.get('years', 0) for exp in experience)
            parts.append(f"{years} years of applied ML experience at product companies")
        
        if behavioral_score > 0.3:
            parts.append("High platform engagement with verified skills")
        
        if not parts:
            if skills:
                parts.append(f"Candidate brings {', '.join(skills[:2])} expertise")
            else:
                parts.append("Candidate demonstrates relevant ML engineering experience")
        
        reasoning = ". ".join(parts[:2]) + "."
        
        if len(reasoning) > 200:
            reasoning = reasoning[:197] + "..."
        
        return reasoning
    
    def rank_pipeline(self, candidate_generator, top_k: int = 100) -> List[RankedCandidate]:
        """
        Execute the complete multistage ranking pipeline.
        
        Args:
            candidate_generator: Generator yielding candidate dictionaries
            top_k: Number of top candidates to output
            
        Returns:
            List of top-k RankedCandidate objects
        """
        print("  [Stage 1] Lexical/BM25 filtering...")
        stage1_candidates = self.stage1_lexical_filter(candidate_generator)
        print(f"  → Stage 1 complete: {len(stage1_candidates)} candidates")
        
        print("  [Stage 2] Semantic embedding matching...")
        stage2_candidates = self.stage2_semantic_match(stage1_candidates)
        print(f"  → Stage 2 complete: {len(stage2_candidates)} candidates")
        
        print("  [Stage 3] Behavioral signal scoring...")
        final_ranked = self.stage3_behavioral_scoring(stage2_candidates, top_k)
        print(f"  → Stage 3 complete: {len(final_ranked)} candidates")
        
        return final_ranked
    
