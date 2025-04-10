#!/usr/bin/env python3
"""
Main speech analysis module with FAISS indexing and gender-neutral feature matching.
This script provides a command-line interface to the speech analysis modules.
"""

import argparse
import os
import sys
import pathlib

# Add the current directory to the path for module imports
script_dir = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, str(script_dir))

# Now import from modules
from modules.feature_extraction import print_feature_indices
from modules.analysis import analyze_speech_comprehensive, find_similar_speeches, test_gender_neutral_matching


def main():
    """Main function for the speech analysis command-line tool."""
    parser = argparse.ArgumentParser(
        description='Speech Analysis with FAISS Indexing and Gender-Neutral Matching'
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Add command: Analyze speech
    analyze_parser = subparsers.add_parser('analyze', help='Analyze speech and add to index')
    analyze_parser.add_argument('--file', type=str, required=True, help='Path to audio file')
    analyze_parser.add_argument('--db', type=str, default='speech_features_db', help='Database directory')
    analyze_parser.add_argument('--description', type=str, help='Description for the speech')
    analyze_parser.add_argument('--skip-index', action='store_true', help='Skip adding to index')
    analyze_parser.add_argument('--new-index', action='store_true', help='Delete existing indices and create a new one')
    
    # Add command: Search for similar speeches
    search_parser = subparsers.add_parser('search', help='Find similar speeches')
    search_parser.add_argument('--file', type=str, required=True, help='Path to audio file')
    search_parser.add_argument('--db', type=str, default='speech_features_db', help='Database directory')
    search_parser.add_argument('--k', type=int, default=5, help='Number of results')
    search_parser.add_argument('--standard', action='store_true', 
                              help='Use standard search instead of gender-neutral')
    search_parser.add_argument('--detailed', action='store_true',
                              help='Show detailed feature comparisons')
    
    # Add command: Test gender-neutral matching
    test_parser = subparsers.add_parser('test', help='Test gender-neutral matching')
    test_parser.add_argument('--db', type=str, default='speech_features_db', help='Database directory')
    
    # Add command: Print feature indices
    features_parser = subparsers.add_parser('features', help='Print feature indices')
    features_parser.add_argument('--file', type=str, required=True, help='Path to audio file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Make sure we have absolute paths
    if args.command in ['analyze', 'search', 'features'] and args.file:
        args.file = os.path.abspath(args.file)
        print(f"Using audio file: {args.file}")
        
    if args.command in ['analyze', 'search', 'test'] and args.db:
        args.db = os.path.abspath(args.db)
        print(f"Using database path: {args.db}")
        
        # Make sure the database directory exists
        os.makedirs(args.db, exist_ok=True)
        print(f"Ensured database directory exists: {args.db}")
    
    # Execute appropriate command
    try:
        if args.command == 'analyze':
            # Create a new index if requested
            if args.new_index and os.path.exists(args.db):
                import shutil
                print(f"Creating new index: removing existing database at {args.db}")
                # Remove the directory and its contents
                shutil.rmtree(args.db)
                print(f"Previous database removed, will create a fresh index.")
            
            analyze_speech_comprehensive(
                args.file, 
                db_path=args.db, 
                description=args.description,
                add_to_index=not args.skip_index
            )
        elif args.command == 'search':
            find_similar_speeches(
                args.file, 
                db_path=args.db, 
                k=args.k, 
                gender_neutral=not args.standard,
                detailed=args.detailed
            )
        elif args.command == 'test':
            test_gender_neutral_matching(db_path=args.db)
        elif args.command == 'features':
            print_feature_indices(args.file)
        else:
            parser.print_help()
    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        print("An error occurred during execution. See details above.")


if __name__ == "__main__":
    main()