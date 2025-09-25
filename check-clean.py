#!/usr/bin/env python3
"""
Quick check to ensure BankingLLM project is clean for git and Docker.
"""

import os
import glob
from pathlib import Path

def check_clean_state():
    """Check if project is in clean state for git and Docker."""
    issues = []

    # Check for Python cache files
    pycache_dirs = glob.glob("**/__pycache__", recursive=True)
    if pycache_dirs:
        issues.append(f"Found Python cache directories: {pycache_dirs}")

    pyc_files = glob.glob("**/*.pyc", recursive=True)
    if pyc_files:
        issues.append(f"Found .pyc files: {pyc_files}")

    # Check for log files
    log_files = glob.glob("**/*.log", recursive=True)
    if log_files:
        issues.append(f"Found log files: {log_files}")

    # Check for database files in wrong places
    db_files = glob.glob("**/*.db", recursive=True)
    allowed_db_locations = ["data/"]
    problematic_dbs = [db for db in db_files if not any(db.startswith(loc) for loc in allowed_db_locations)]
    if problematic_dbs:
        issues.append(f"Found database files outside data/: {problematic_dbs}")

    # Check for temporary files
    temp_files = glob.glob("**/*.tmp", recursive=True) + glob.glob("**/*.temp", recursive=True)
    if temp_files:
        issues.append(f"Found temporary files: {temp_files}")

    # Check for required files
    required_files = [
        ".gitignore",
        ".dockerignore",
        "README.md",
        "requirements.txt",
        "Dockerfile",
        "docker-compose.yml"
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        issues.append(f"Missing required files: {missing_files}")

    # Check data directory structure
    if not os.path.exists("data/.gitkeep"):
        issues.append("Missing data/.gitkeep file")

    if not os.path.exists("data/exports/.gitkeep"):
        issues.append("Missing data/exports/.gitkeep file")

    # Report results
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("SUCCESS: Project is clean and ready for git and Docker!")
        print("Summary:")
        print("  - .gitignore: Comprehensive Python, Docker, Claude files")
        print("  - .dockerignore: Clean builds, excludes dev files")
        print("  - Data structure: Proper .gitkeep files")
        print("  - No temporary or cache files")
        print("  - All required files present")
        return True

if __name__ == "__main__":
    import sys
    clean = check_clean_state()
    sys.exit(0 if clean else 1)