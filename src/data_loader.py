"""
Data loader module for streaming and parsing candidate data.
Optimized for memory efficiency with chunked processing of large datasets.
"""

import gzip
import json
from typing import Dict, List, Iterator, Optional
from pathlib import Path
import jsonlines


class Candidate:
    """Lightweight candidate data structure for fast access."""
    
    def __init__(self, data: Dict):
        self.candidate_id = data.get('candidate_id', '')
        self.name = data.get('name', '')
        self.email = data.get('email', '')
        self.skills = data.get('skills', [])
        self.experience = data.get('experience', [])
        self.projects = data.get('projects', [])
        self.education = data.get('education', [])
        self.location = data.get('location', {})
        self.redrob_signals = data.get('redrob_signals', {})
        self.raw_data = data
    
    def to_dict(self) -> Dict:
        """Convert back to dictionary format."""
        return self.raw_data


class CandidateDataLoader:
    """
    Efficient data loader for streaming candidates from .jsonl.gz files.
    Implements chunked reading to stay within 16GB RAM constraints.
    """
    
    def __init__(self, file_path: str, chunk_size: int = 1000):
        """
        Initialize the data loader.
        
        Args:
            file_path: Path to the candidates.jsonl.gz file
            chunk_size: Number of candidates to load per chunk
        """
        self.file_path = Path(file_path)
        self.chunk_size = chunk_size
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Candidate file not found: {file_path}")
    
    def stream_candidates(self) -> Iterator[Candidate]:
        """
        Stream candidates one at a time from the compressed file.
        Memory-efficient generator for processing 100k+ candidates.
        
        Yields:
            Candidate objects one at a time
        """
        if self.file_path.suffix == '.gz':
            with gzip.open(self.file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        yield Candidate(data)
        else:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        yield Candidate(data)
    
    def stream_chunks(self) -> Iterator[List[Candidate]]:
        """
        Stream candidates in chunks for batched processing.
        
        Yields:
            List of Candidate objects (chunk_size per batch)
        """
        chunk = []
        for candidate in self.stream_candidates():
            chunk.append(candidate)
            if len(chunk) >= self.chunk_size:
                yield chunk
                chunk = []
        
        if chunk:  # Yield remaining candidates
            yield chunk
    
    def load_sample(self, n: int = 100) -> List[Candidate]:
        """
        Load a sample of candidates for testing/schema inspection.
        
        Args:
            n: Number of candidates to sample
            
        Returns:
            List of Candidate objects
        """
        candidates = []
        for i, candidate in enumerate(self.stream_candidates()):
            if i >= n:
                break
            candidates.append(candidate)
        return candidates
    
    def count_total(self) -> int:
        """
        Count total number of candidates in the file.
        
        Returns:
            Total count of candidates
        """
        count = 0
        for _ in self.stream_candidates():
            count += 1
        return count


class JobDescriptionParser:
    """Parser for extracting structured information from job descriptions."""
    
    def __init__(self, jd_path: str):
        """
        Initialize the JD parser.
        
        Args:
            jd_path: Path to the job_description.md file
        """
        self.jd_path = Path(jd_path)
        if not self.jd_path.exists():
            raise FileNotFoundError(f"Job description not found: {jd_path}")
        
        self.content = self.jd_path.read_text(encoding='utf-8')
    
    def parse(self) -> Dict:
        """
        Extract structured information from the job description.
        
        Returns:
            Dictionary with extracted fields:
            - title: Job title
            - required_skills: List of required technical skills
            - experience_level: Junior/Mid/Senior/Lead
            - min_experience_years: Minimum years of experience
            - responsibilities: List of key responsibilities
            - requirements: List of requirements
            - negative_signals: List of trap/negative signals to avoid
        """
        result = {
            'title': '',
            'required_skills': [],
            'experience_level': 'Mid',
            'min_experience_years': 0,
            'responsibilities': [],
            'requirements': [],
            'negative_signals': [],
            'full_text': self.content
        }
        
        lines = self.content.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Detect section headers
            if line.lower().startswith('#'):
                current_section = line.lstrip('#').strip().lower()
                if 'title' in current_section or 'role' in current_section:
                    result['title'] = line.lstrip('#').strip()
                continue
            
            # Extract skills (common patterns)
            if any(keyword in line.lower() for keyword in ['skill', 'technology', 'tech stack', 'required']):
                if ':' in line or '-' in line:
                    skills = self._extract_skills_from_line(line)
                    result['required_skills'].extend(skills)
            
            # Extract experience level
            if 'senior' in line.lower():
                result['experience_level'] = 'Senior'
            elif 'junior' in line.lower():
                result['experience_level'] = 'Junior'
            elif 'lead' in line.lower() or 'principal' in line.lower():
                result['experience_level'] = 'Lead'
            
            # Extract years of experience
            if 'year' in line.lower():
                years = self._extract_years(line)
                if years > result['min_experience_years']:
                    result['min_experience_years'] = years
            
            # Extract negative/trap signals
            if any(keyword in line.lower() for keyword in ['avoid', 'not', 'trap', 'warning', 'red flag']):
                result['negative_signals'].append(line)
        
        # Deduplicate skills
        result['required_skills'] = list(set(result['required_skills']))
        
        return result
    
    def _extract_skills_from_line(self, line: str) -> List[str]:
        """Extract skill names from a line of text."""
        # Remove common prefixes
        for prefix in ['Skills:', 'Required:', 'Technologies:', 'Tech Stack:', '-']:
            line = line.replace(prefix, '').strip()
        
        # Split by common delimiters
        skills = []
        for delimiter in [',', ';', '|', '•', '-']:
            if delimiter in line:
                skills = [s.strip() for s in line.split(delimiter)]
                break
        
        if not skills:
            skills = [line]
        
        return [s for s in skills if s and len(s) > 1]
    
    def _extract_years(self, line: str) -> int:
        """Extract number of years from a line."""
        import re
        matches = re.findall(r'(\d+)\+?\s*years?', line.lower())
        if matches:
            return int(matches[0])
        return 0
