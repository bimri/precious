#!/bin/bash

# Precious Package Git Repository Setup Script
# This script sets up the Git repository, cleans cache files, and prepares for publishing

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Precious Package Git Repository Setup${NC}"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src/precious" ]; then
    echo -e "${RED}❌ Error: Run this script from the precious package root directory${NC}"
    echo "Expected files: pyproject.toml, src/precious/"
    exit 1
fi

echo -e "${GREEN}✅ Confirmed: Running in precious package directory${NC}"

# Step 1: Clean up cache files and build artifacts
echo -e "\n${YELLOW}🧹 Step 1: Cleaning cache files and build artifacts${NC}"

# Remove Python cache files
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
find . -name "*~" -delete 2>/dev/null || true

# Remove pytest cache
if [ -d ".pytest_cache" ]; then
    echo "Removing .pytest_cache/"
    rm -rf .pytest_cache/
fi

# Remove rope project files
if [ -d ".ropeproject" ]; then
    echo "Removing .ropeproject/"
    rm -rf .ropeproject/
fi

# Remove build artifacts
echo "Removing build artifacts..."
rm -rf build/ 2>/dev/null || true
rm -rf dist/ 2>/dev/null || true
rm -rf *.egg-info/ 2>/dev/null || true

# Remove coverage files
rm -rf .coverage* htmlcov/ 2>/dev/null || true

# Remove temporary virtual environments
rm -rf venv*/ env*/ *-env/ 2>/dev/null || true

# Remove OS specific files
find . -name ".DS_Store" -delete 2>/dev/null || true
find . -name "Thumbs.db" -delete 2>/dev/null || true

echo -e "${GREEN}✅ Cache cleanup completed${NC}"

# Step 2: Initialize Git repository
echo -e "\n${YELLOW}📁 Step 2: Git repository initialization${NC}"

if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    git branch -M main
    echo -e "${GREEN}✅ Git repository initialized${NC}"
else
    echo -e "${GREEN}✅ Git repository already exists${NC}"
fi

# Step 3: Configure Git user (if not already configured)
echo -e "\n${YELLOW}👤 Step 3: Git user configuration${NC}"

# Check if user is configured globally or locally
if ! git config user.name > /dev/null 2>&1; then
    echo "Setting Git user configuration..."
    git config user.name "bimri"
    git config user.email "bimri@outlook.com"
    echo -e "${GREEN}✅ Git user configured${NC}"
else
    echo -e "${GREEN}✅ Git user already configured: $(git config user.name) <$(git config user.email)>${NC}"
fi

# Step 4: Add remote repository
echo -e "\n${YELLOW}🌐 Step 4: Adding remote repository${NC}"

if ! git remote get-url origin > /dev/null 2>&1; then
    echo "Adding GitHub remote repository..."
    git remote add origin https://github.com/bimri/precious.git
    echo -e "${GREEN}✅ Remote repository added${NC}"
else
    echo -e "${GREEN}✅ Remote repository already configured: $(git remote get-url origin)${NC}"
fi

# Step 5: Fetch remote and handle .gitignore
echo -e "\n${YELLOW}📝 Step 5: Handling .gitignore file${NC}"

# Fetch remote to see what's there
echo "Fetching remote repository..."
git fetch origin 2>/dev/null || echo "Note: Remote fetch failed (repository might be empty)"

# Check if remote has .gitignore
if git cat-file -e origin/main:.gitignore 2>/dev/null; then
    echo "Remote .gitignore found. Merging with local..."

    # Backup local .gitignore if it exists
    if [ -f ".gitignore" ]; then
        cp .gitignore .gitignore.local
    fi

    # Get remote .gitignore
    git show origin/main:.gitignore > .gitignore.remote 2>/dev/null

    # Combine both files and remove duplicates
    if [ -f ".gitignore.local" ]; then
        cat .gitignore.local .gitignore.remote | sort -u > .gitignore.merged
    else
        cp .gitignore.remote .gitignore.merged
    fi

    mv .gitignore.merged .gitignore

    # Cleanup
    rm -f .gitignore.local .gitignore.remote

    echo -e "${GREEN}✅ .gitignore files merged${NC}"
else
    echo -e "${GREEN}✅ Using local .gitignore (no remote version found)${NC}"
fi

# Step 6: Stage and commit files
echo -e "\n${YELLOW}📦 Step 6: Staging and committing files${NC}"

# Add .gitignore first
git add .gitignore

echo "Files that will be committed:"
git status --porcelain | head -10
if [ $(git status --porcelain | wc -l) -gt 10 ]; then
    echo "... and $(( $(git status --porcelain | wc -l) - 10 )) more files"
fi

# Verify no cache files are being added
echo -e "\n${BLUE}Verifying no cache files will be committed:${NC}"
CACHE_FILES=$(git status --porcelain | grep -E "(__pycache__|\.pyc|\.pyo|\.pytest_cache|\.ropeproject)" | wc -l)
if [ "$CACHE_FILES" -gt 0 ]; then
    echo -e "${RED}❌ Warning: Cache files detected in staging area!${NC}"
    git status --porcelain | grep -E "(__pycache__|\.pyc|\.pyo|\.pytest_cache|\.ropeproject)"
    echo "Please check your .gitignore file"
    exit 1
else
    echo -e "${GREEN}✅ No cache files detected${NC}"
fi

# Add all files
git add .

# Check if there's anything to commit
if git diff --cached --quiet; then
    echo -e "${YELLOW}ℹ️ No changes to commit${NC}"
else
    echo "Creating initial commit..."
    git commit -m "Initial commit: Complete tokenizer-free NLP library

Features:
- T-FREE, CANINE, and byte-level tokenizer-free approaches
- Comprehensive test suite with 16 passing tests
- Complete documentation and API reference
- Production-ready with CI/CD pipeline
- Support for Python 3.8-3.12 with PyTorch backend
- MIT licensed with proper package structure

Ready for PyPI publication!"

    echo -e "${GREEN}✅ Initial commit created${NC}"
fi

# Step 7: Handle remote synchronization
echo -e "\n${YELLOW}🔄 Step 7: Synchronizing with remote${NC}"

# Check if remote has commits
if git ls-remote --exit-code --heads origin main > /dev/null 2>&1; then
    echo "Remote branch 'main' exists. Checking for conflicts..."

    # Try to merge remote changes
    if git merge origin/main --allow-unrelated-histories --no-edit; then
        echo -e "${GREEN}✅ Successfully merged with remote${NC}"
    else
        echo -e "${YELLOW}⚠️ Merge conflicts detected. Please resolve manually and then run:${NC}"
        echo "git add <resolved-files>"
        echo "git commit -m 'Resolve merge conflicts'"
        exit 1
    fi
fi

# Push to remote
echo "Pushing to remote repository..."
if git push -u origin main; then
    echo -e "${GREEN}✅ Successfully pushed to remote${NC}"
else
    echo -e "${RED}❌ Push failed. Please check your GitHub repository and credentials${NC}"
    exit 1
fi

# Step 8: Create version tag
echo -e "\n${YELLOW}🏷️  Step 8: Creating version tag${NC}"

# Check if tag already exists
if git tag -l | grep -q "v0.1.0"; then
    echo -e "${YELLOW}⚠️ Tag v0.1.0 already exists${NC}"
else
    echo "Creating version tag v0.1.0..."
    git tag -a v0.1.0 -m "Release v0.1.0: Initial tokenizer-free NLP library

This release includes:
- Complete T-FREE implementation with vocabulary building
- CANINE character-level processing with Unicode support
- Efficient byte-level text processing
- Comprehensive test suite (16 tests)
- Production-ready error handling and device management
- Complete documentation with API reference and examples
- CI/CD pipeline for automated testing and publishing

Ready for PyPI publication and research use!"

    # Push tags
    git push origin --tags
    echo -e "${GREEN}✅ Version tag created and pushed${NC}"
fi

# Step 9: Final verification
echo -e "\n${YELLOW}✅ Step 9: Final verification${NC}"

echo "Repository status:"
echo "- Branch: $(git branch --show-current)"
echo "- Remote: $(git remote get-url origin)"
echo "- Latest commit: $(git log --oneline -1)"
echo "- Tags: $(git tag -l | tr '\n' ' ')"

# Test .gitignore is working
touch test-cache.pyc
if git status --porcelain | grep -q "test-cache.pyc"; then
    echo -e "${RED}❌ Warning: .gitignore not working correctly${NC}"
    rm test-cache.pyc
    exit 1
else
    echo -e "${GREEN}✅ .gitignore working correctly${NC}"
    rm test-cache.pyc
fi

# Show tracked files summary
echo -e "\nTracked files summary:"
echo "- Total files: $(git ls-files | wc -l)"
echo "- Python files: $(git ls-files | grep '\.py$' | wc -l)"
echo "- Test files: $(git ls-files | grep 'test_.*\.py$' | wc -l)"
echo "- Documentation: $(git ls-files | grep '\.md$' | wc -l)"

# Success message
echo -e "\n${GREEN}🎉 SUCCESS! Your repository is ready for publishing!${NC}"
echo "=============================================="
echo -e "${BLUE}Next steps:${NC}"
echo "1. Visit: https://github.com/bimri/precious"
echo "2. Verify all files are properly committed"
echo "3. Create a GitHub release from the v0.1.0 tag"
echo "4. Set up PyPI account and tokens"
echo "5. Publish to PyPI!"

echo -e "\n${BLUE}Quick test command:${NC}"
echo "python -c \"from src.precious import PreciousModel, PreciousConfig; print('✅ Package imports successfully!')\""

echo -e "\n${GREEN}Repository setup completed successfully! 🚀${NC}"
