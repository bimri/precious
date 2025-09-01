# Git Setup Commands for Precious Package

## Overview
This file contains the exact commands to properly set up your Git repository, merge with the remote Python .gitignore, and prepare for publishing.

## Step 1: Check Current Git Status
```bash
cd /home/bimri/Desktop/precious

# Check if git is already initialized
git status

# If not initialized, you'll see: "fatal: not a git repository"
# If initialized, you'll see the current status
```

## Step 2: Initialize and Configure Git (if needed)
```bash
# Initialize git repository
git init

# Configure git user (replace with your info)
git config user.name "bimri"
git config user.email "bimri@outlook.com"

# Set default branch to main
git branch -M main
```

## Step 3: Add Remote Repository
```bash
# Add your GitHub repository as remote
git remote add origin https://github.com/bimri/precious.git

# Verify remote was added
git remote -v
```

## Step 4: Fetch Remote .gitignore and Merge
```bash
# Fetch remote repository content
git fetch origin

# Check what files exist on remote
git ls-tree origin/main

# If remote has .gitignore, merge it with local
# Option A: If you want to keep both (recommended)
git checkout origin/main -- .gitignore
# This will overwrite local .gitignore with remote version

# Option B: Manual merge (if you want to combine both)
# First, backup your local .gitignore
cp .gitignore .gitignore.local

# Get remote .gitignore
git show origin/main:.gitignore > .gitignore.remote

# Combine both files (remove duplicates)
cat .gitignore.local .gitignore.remote | sort -u > .gitignore.combined
mv .gitignore.combined .gitignore

# Clean up
rm .gitignore.local .gitignore.remote
```

## Step 5: Clean Up Cache Files Before First Commit
```bash
# Remove existing cache directories and files
rm -rf .pytest_cache/
rm -rf .ropeproject/
rm -rf __pycache__/
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name "*~" -delete
find . -name ".DS_Store" -delete

# Remove any build artifacts
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
rm -rf .coverage*
rm -rf htmlcov/

# Remove any virtual environments in project directory
rm -rf venv*/
rm -rf env*/
rm -rf *-env/
```

## Step 6: Stage and Commit Files
```bash
# Add .gitignore first
git add .gitignore

# Check what files will be committed (verify no cache files)
git status

# Add all project files (cache files should be ignored)
git add .

# Verify what's staged (should NOT include cache files)
git status

# Create initial commit
git commit -m "Initial commit: Complete tokenizer-free NLP library

Features:
- T-FREE, CANINE, and byte-level tokenizer-free approaches
- Comprehensive test suite with 16 passing tests
- Complete documentation and API reference
- Production-ready with CI/CD pipeline
- Support for Python 3.8-3.12 with PyTorch backend
- MIT licensed with proper package structure"
```

## Step 7: Handle Remote Conflicts (if any)
```bash
# If remote has commits, you need to merge or rebase
# Check if remote has commits
git fetch origin
git log origin/main --oneline

# Option A: Merge (creates merge commit)
git merge origin/main --allow-unrelated-histories

# Option B: Rebase (cleaner history)
git rebase origin/main

# If conflicts occur, resolve them and continue
# git status  # shows conflicted files
# Edit conflicted files manually
# git add <resolved-files>
# git rebase --continue  # or git merge --continue
```

## Step 8: Push to Remote
```bash
# Push main branch to remote
git push -u origin main

# Verify push was successful
git log --oneline -5
```

## Step 9: Create Version Tag
```bash
# Create annotated tag for version 0.1.0
git tag -a v0.1.0 -m "Release v0.1.0: Initial tokenizer-free NLP library

This release includes:
- Complete T-FREE implementation with vocabulary building
- CANINE character-level processing with Unicode support
- Efficient byte-level text processing
- Comprehensive test suite (16 tests)
- Production-ready error handling and device management
- Complete documentation with API reference and examples
- CI/CD pipeline for automated testing and publishing"

# Push tags to remote
git push origin --tags

# Verify tag was created
git tag -l
git show v0.1.0
```

## Alternative: One-Command Setup (if starting fresh)
```bash
cd /home/bimri/Desktop/precious

# Complete setup in one go
git init
git config user.name "bimri"
git config user.email "bimri@outlook.com"
git branch -M main
git remote add origin https://github.com/bimri/precious.git

# Clean cache files
rm -rf .pytest_cache/ .ropeproject/ __pycache__/ build/ dist/ *.egg-info/
find . -name "*.pyc" -delete

# Fetch and merge remote
git fetch origin
git checkout origin/main -- .gitignore || echo "No remote .gitignore found"

# Commit everything
git add .
git commit -m "Initial commit: Complete tokenizer-free NLP library"

# Handle remote merge if needed
git pull origin main --allow-unrelated-histories || git push -u origin main

# Create version tag
git tag -a v0.1.0 -m "Release v0.1.0: Initial tokenizer-free NLP library"
git push origin --tags
```

## Verification Commands
```bash
# Verify repository status
git status                    # Should be clean
git remote -v                 # Should show origin
git branch -a                 # Should show main and origin/main
git tag -l                    # Should show v0.1.0
git log --oneline -5          # Should show recent commits

# Verify .gitignore is working
touch test-cache.pyc
git status                    # Should NOT show test-cache.pyc
rm test-cache.pyc

# Check what files are tracked
git ls-files | head -20       # Should show source files, not cache files
```

## Common Issues and Solutions

### Issue 1: "fatal: refusing to merge unrelated histories"
```bash
# Solution: Use --allow-unrelated-histories
git pull origin main --allow-unrelated-histories
```

### Issue 2: Remote .gitignore conflicts with local
```bash
# Solution: Manually merge both .gitignore files
git show origin/main:.gitignore > .gitignore.remote
cat .gitignore .gitignore.remote | sort -u > .gitignore.merged
mv .gitignore.merged .gitignore
rm .gitignore.remote
```

### Issue 3: Cache files already committed
```bash
# Solution: Remove from git tracking but keep locally
git rm --cached -r .pytest_cache/
git rm --cached -r .ropeproject/
find . -name "*.pyc" -exec git rm --cached {} \;
git commit -m "Remove cache files from tracking"
```

### Issue 4: Large file errors
```bash
# Solution: Use Git LFS for large files (if any)
git lfs install
git lfs track "*.pth"
git lfs track "*.model"
git add .gitattributes
```

## Files That Should Be Tracked
✅ src/precious/*.py
✅ tests/*.py
✅ docs/*.md
✅ pyproject.toml
✅ setup.py
✅ requirements.txt
✅ README.md
✅ LICENSE
✅ CHANGELOG.md
✅ .gitignore
✅ MANIFEST.in

## Files That Should NOT Be Tracked
❌ .pytest_cache/
❌ .ropeproject/
❌ __pycache__/
❌ *.pyc, *.pyo
❌ build/
❌ dist/
❌ *.egg-info/
❌ .coverage*
❌ venv*/
❌ .DS_Store

## Ready for Publishing!
After completing these steps, your repository will be:
- ✅ Clean (no cache files)
- ✅ Properly configured with remote
- ✅ Tagged for release
- ✅ Ready for GitHub Actions CI/CD
- ✅ Ready for PyPI publishing

## Next Steps
1. Create GitHub release from v0.1.0 tag
2. Set up PyPI account and tokens
3. Configure GitHub Actions secrets
4. Publish to PyPI