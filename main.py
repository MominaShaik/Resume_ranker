"""
Main CLI entry point for the Redrob Candidate Ranking System.
Orchestrates the complete pipeline from data loading to final CSV output.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional
import pandas as pd

from src.data_loader import CandidateDataLoader, JobDescriptionParser
from src.ranker import MultistageRanker


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Redrob Candidate Ranking System - Intelligent Candidate Discovery'
    )
    
    parser.add_argument(
        '--jd',
        type=str,
        default='data/job_description.md',
        help='Path to job description markdown file'
    )
    
    parser.add_argument(
        '--candidates',
        type=str,
        default='data/candidates.jsonl.gz',
        help='Path to candidates JSONL.gz file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output/ranked_candidates.csv',
        help='Path to output CSV file'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=100,
        help='Number of top candidates to output'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence transformer model to use (default: all-MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--semantic-weight',
        type=float,
        default=0.4,
        help='Weight for semantic score (default: 0.4)'
    )
    
    parser.add_argument(
        '--experience-weight',
        type=float,
        default=0.25,
        help='Weight for experience score (default: 0.25)'
    )
    
    parser.add_argument(
        '--behavioral-weight',
        type=float,
        default=0.35,
        help='Weight for behavioral score (default: 0.35)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate output after generation'
    )
    
    return parser.parse_args()


def main():
    """Main execution pipeline."""
    args = parse_arguments()
    
    print("=" * 60)
    print("Redrob Candidate Ranking System")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Load and parse job description
        print("\n[1/5] Loading job description...")
        jd_parser = JobDescriptionParser(args.jd)
        jd_requirements = jd_parser.parse()
        
        print(f"  - Job Title: {jd_requirements.get('title', 'N/A')}")
        print(f"  - Required Skills: {len(jd_requirements.get('required_skills', []))}")
        print(f"  - Experience Level: {jd_requirements.get('experience_level', 'N/A')}")
        print(f"  - Min Years: {jd_requirements.get('min_experience_years', 0)}")
        
        # Step 2: Initialize data loader
        print("\n[2/5] Initializing candidate data loader...")
        data_loader = CandidateDataLoader(args.candidates, chunk_size=1000)
        
        # Count total candidates (optional - can be slow for large files)
        print("  - Counting total candidates...")
        total_candidates = data_loader.count_total()
        print(f"  - Total candidates: {total_candidates}")
        
        # Step 3: Initialize multistage ranker
        print("\n[3/5] Initializing multistage ranking pipeline...")
        ranker = MultistageRanker(
            semantic_weight=args.semantic_weight,
            experience_weight=args.experience_weight,
            behavioral_weight=args.behavioral_weight,
            stage1_target=2000,
            stage2_target=500
        )
        
        ranker.set_job_description(jd_requirements)
        print("  - Multistage pipeline initialized")
        
        # Step 4: Execute multistage ranking pipeline
        print("\n[4/5] Executing multistage ranking pipeline...")
        
        # Stream candidates through the complete pipeline
        ranked = ranker.rank_pipeline(data_loader.stream_candidates(), top_k=args.top_k)
        
        # Step 5: Extract top-k and generate output
        print(f"\n[5/5] Extracting top {args.top_k} candidates...")
        top_ranked = ranked
        
        # Prepare output data
        output_data = []
        for i, candidate in enumerate(top_ranked):
            output_data.append({
                'candidate_id': candidate.candidate_id,
                'rank': i + 1,
                'score': round(candidate.final_score, 4),
                'reasoning': candidate.reasoning
            })
        
        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write CSV
        df = pd.DataFrame(output_data)
        df.to_csv(output_path, index=False)
        
        print(f"  - Output saved to: {output_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("RANKING COMPLETE")
        print("=" * 60)
        print(f"Total candidates processed: {total_candidates}")
        print(f"Top {args.top_k} candidates output")
        print(f"Output file: {output_path}")
        
        # Print top 5 for preview
        print("\nTop 5 Candidates:")
        print("-" * 60)
        for i, row in df.head(5).iterrows():
            print(f"#{int(row['rank'])} | ID: {row['candidate_id']} | Score: {row['score']}")
            print(f"     Reasoning: {row['reasoning']}")
            print()
        
        # Validation
        if args.validate:
            print("\nValidating output...")
            try:
                from validate_submission import validate_submission
                is_valid, errors = validate_submission(output_path)
                if is_valid:
                    print("  ✓ Output validation passed")
                else:
                    print("  ✗ Output validation failed:")
                    for error in errors:
                        print(f"    - {error}")
            except ImportError:
                print("  - Validator not found, skipping validation")
        
        # Performance metrics
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
        print(f"Average time per candidate: {elapsed_time / total_candidates * 1000:.2f} ms")
        
        if elapsed_time < 300:
            print("✓ Within 5-minute constraint")
        else:
            print("⚠ Exceeded 5-minute constraint")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
