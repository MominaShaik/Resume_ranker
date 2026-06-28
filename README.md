# Redrob Candidate Ranking System

An intelligent candidate discovery and ranking AI system for the Redrob Hackathon. This system moves beyond rigid keyword matching to implement semantic, behavioral-aware ranking that evaluates candidates like an elite human recruiter.

## System Overview

The system implements a **multistage cascading funnel architecture** for efficient 100K candidate processing:

```
Stage 1: Ultra-Fast Lexical/BM25 Filter (100K → ~2K)
  ↓
Stage 2: Semantic Embedding Matching (~2K → ~500)
  ↓
Stage 3: Behavioral Signal Scoring with Multipliers (~500 → Top 100)
```

### Scoring Formula

```
Final Score = (Semantic × Experience × Behavioral × Multiplier) - Hard Penalty
```

### Key Features

- **Multistage Pipeline**: Cascading funnel reduces expensive operations by 98%
- **Memory-Efficient Streaming**: Line-by-line processing, never loads full dataset
- **CPU-Optimized**: Lightweight sentence-transformers (all-MiniLM-L6-v2) for fast inference
- **Dynamic Multipliers**: Boosts high-intent candidates (response rate, verified skills, recent activity)
- **Hard Penalties**: Eliminates trap candidates (job hoppers, bots, fraud)
- **Redrob-Specific Logic**: Product company experience, ranking/retrieval systems, behavioral signals
- **Offline Execution**: No API calls during inference - completely local processing
- **5-Minute Constraint**: Optimized to complete ranking within strict time limits
- **Premium Dashboard**: Interactive Streamlit web interface for result visualization

## Project Structure

```
Resume_ranker/
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Streaming and parsing of candidate data
│   ├── ranker.py           # Multistage cascading funnel ranking logic
│   └── utils.py            # Text preprocessing, embeddings, BM25, behavioral scoring
├── data/
│   ├── job_description.md  # Job requirements (input)
│   └── candidates.jsonl.gz # Candidate database (input)
├── output/
│   └── ranked_candidates.csv # Final ranked output
├── main.py                 # CLI entry point for ranking pipeline
├── app.py                  # Streamlit dashboard for visualization
├── validate_submission.py  # Output validation
├── requirements.txt        # CPU-optimized dependencies
└── README.md              # This file
```

## Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Resume_ranker
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Run Ranking Pipeline

```bash
python main.py --jd data/job_description.md --candidates data/candidates.jsonl.gz --validate
```

### Advanced Options

```bash
python main.py \
  --jd data/job_description.md \
  --candidates data/candidates.jsonl.gz \
  --output output/ranked_candidates.csv \
  --top-k 100 \
  --semantic-weight 0.35 \
  --experience-weight 0.25 \
  --behavioral-weight 0.25 \
  --validate
```

### Command-Line Arguments

- `--jd`: Path to job description markdown file (default: `data/job_description.md`)
- `--candidates`: Path to candidates JSONL.gz file (default: `data/candidates.jsonl.gz`)
- `--output`: Path to output CSV file (default: `output/ranked_candidates.csv`)
- `--top-k`: Number of top candidates to output (default: `100`)
- `--semantic-weight`: Weight for semantic score (default: `0.35`)
- `--experience-weight`: Weight for experience score (default: `0.25`)
- `--behavioral-weight`: Weight for behavioral score (default: `0.25`)
- `--validate`: Validate output after generation

### Launch Dashboard

After running the ranking pipeline, launch the premium Streamlit dashboard:

```bash
python -m streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### Validation

Validate the output CSV independently:

```bash
python validate_submission.py output/ranked_candidates.csv
```

## Data Format

### Input: Job Description (Markdown)

The job description should be a markdown file containing:
- Job title and role
- Required skills and technologies
- Experience level and requirements
- Responsibilities
- Any negative signals or "traps" to avoid

Example:
```markdown
# Senior Machine Learning Engineer

## Required Skills
- Python, TensorFlow, PyTorch
- NLP and Computer Vision
- MLOps and deployment

## Requirements
- 5+ years of experience
- PhD or MS in CS/ML preferred
```

### Input: Candidates (JSONL.gz)

Each line should be a JSON object with:
```json
{
  "candidate_id": "unique_id",
  "name": "Candidate Name",
  "email": "email@example.com",
  "skills": ["Python", "TensorFlow", "NLP"],
  "experience": [
    {
      "title": "ML Engineer",
      "company": "Tech Corp",
      "years": 3,
      "description": "Built NLP models"
    }
  ],
  "projects": [...],
  "education": [...],
  "location": {...},
  "redrob_signals": {
    "profile_completion": 85,
    "verified_skills": true,
    "response_rate": 90,
    "last_active_days": 5,
    ...
  }
}
```

### Output: Ranked Candidates (CSV)

The output CSV must contain:
- `candidate_id`: Unique identifier
- `rank`: Position in ranking (1-100)
- `score`: Composite score (0-1 range)
- `reasoning`: 1-2 sentence human-readable justification

Example:
```csv
candidate_id,rank,score,reasoning
candidate_001,1,0.8745,Strong match with expertise in Python and TensorFlow. Proven track record at Tech Corp as ML Engineer with high platform engagement.
candidate_002,2,0.8521,Strong alignment with role requirements. Demonstrates relevant professional experience in NLP and computer vision.
```

## Behavioral Signals

The system evaluates 23 behavioral signals with dynamic multipliers and hard penalties:

**Positive Signals (Boosts):**
- Profile completion
- Verified skills
- Response rate (dynamic multiplier: >80% = 1.1x boost)
- Recent activity (dynamic multiplier for recent logins)
- Application count
- Interview attendance (dynamic multiplier: >90% = boost)
- Offer acceptance rate
- Platform engagement
- Referrals
- Skill endorsements

**Negative Signals (Hard Penalties):**
- Profile mismatch
- Failed verification (hard penalty: -0.8)
- Suspicious activity (hard penalty: -0.7)
- Job hopping frequency (hard penalty: >3 jobs = -0.4)
- Employment gaps
- Location mismatch
- Salary expectation mismatch
- Communication issues
- Negative feedback
- Incomplete profile
- Duplicate profile (hard penalty: -0.9)
- Bot-like behavior (hard penalty: -0.95)

**Redrob-Specific Disqualifiers:**
- Consulting-only background (TCS, Infosys, Wipro, etc.)
- Pure research background (no production deployment)
- Framework enthusiasts (LangChain tutorials without systems thinking)
- Recent LLM-only experience (no pre-LLM ML production)
- No recent code (18+ months without production code)

## Performance Optimization

### Memory Constraints (16GB RAM)
- Streaming: Line-by-line reading of JSONL.gz files
- Multistage funnel: Reduces memory footprint by 98%
- Pre-filtering: Stage 1 reduces 100K to ~2K before expensive operations
- Lightweight models: 384-dim embeddings (all-MiniLM-L6-v2)

### Time Constraints (5 minutes)
- Stage 1: Ultra-fast lexical/BM25 filtering (~0.5s for 100K)
- Stage 2: Semantic embeddings only on ~2K candidates (~30s)
- Stage 3: Behavioral scoring on ~500 candidates (~10s)
- Total: ~60-90 seconds for 100K candidates (well within 5-minute limit)

### Multistage Pipeline Benefits
- **Stage 1**: Eliminates 98% of candidates using cheap heuristics
- **Stage 2**: Expensive embeddings only on viable candidates
- **Stage 3**: Fine-grained behavioral scoring on elite pool
- **Result**: 50-100x speedup vs naive approach

## Dashboard Features

The Streamlit dashboard provides:

- **Premium Dark Mode**: Glowing cards with #0e1117 background
- **Live Metrics**: Total processed, shortlisted finalists, match strength tier
- **Interactive Progress Bars**: Color-filled visual trackers for AI scores
- **Plotly Histogram**: Score distribution density visualization
- **Deep-Dive Inspector**: Dropdown search for detailed candidate analysis
- **AI Reasoning**: Human-like 1-2 sentence analysis for each candidate

## Troubleshooting

### Out of Memory
- Reduce `stage1_target` in `main.py` (default: 2000)
- Reduce `stage2_target` in `main.py` (default: 500)
- Use smaller embedding model

### Slow Execution
- Increase pre-filtering aggressiveness (reduce stage targets)
- Use `all-MiniLM-L6-v2` (already default)
- Reduce `top_k` if needed

### Model Download Issues
- Models download automatically on first run
- Ensure internet access for initial setup
- Models are cached locally for subsequent runs

### Streamlit Not Found
- Use `python -m streamlit run app.py` instead of `streamlit run app.py`

## Citation

If you use this system for the Redrob Hackathon, please reference the project appropriately.

## License

MIT License - See LICENSE file for details
