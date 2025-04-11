#!/bin/bash

# Colors for console output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}====================================${NC}"
echo -e "${BLUE} TabulaPrima Dataset Cache Setup ${NC}"
echo -e "${BLUE}====================================${NC}"

# Check Python version
python_version=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
required_version="3.12"

echo -e "\n${BLUE}Checking Python version...${NC}"
if [ $(echo "$python_version < $required_version" | bc -l) -eq 1 ]; then
    echo -e "${RED}Error: Python $required_version or higher is required. You have Python $python_version.${NC}"
    echo -e "${YELLOW}Please install Python $required_version or higher and try again.${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Python $python_version detected.${NC}"
fi

# Check for virtual environment
echo -e "\n${BLUE}Checking for virtual environment...${NC}"
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}⚠ No virtual environment detected.${NC}"
    echo -e "${YELLOW}It's recommended to install in a virtual environment.${NC}"

    read -p "Continue with installation anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Installation cancelled.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Virtual environment detected: $(basename $VIRTUAL_ENV)${NC}"
fi

# Create necessary directories
echo -e "\n${BLUE}Creating directories...${NC}"
mkdir -p dataset_cache
mkdir -p checkpoints
mkdir -p models
echo -e "${GREEN}✓ Created necessary directories.${NC}"

# Install dependencies
echo -e "\n${BLUE}Installing dependencies...${NC}"
pip install -e .
echo -e "${GREEN}✓ Dependencies installed.${NC}"

# Check for optional dependencies
echo -e "\n${BLUE}Checking for optional dependencies...${NC}"

# Check for tensorboard
if pip show tensorboard > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Tensorboard is installed.${NC}"
else
    echo -e "${YELLOW}⚠ Tensorboard is not installed.${NC}"
    read -p "Would you like to install tensorboard for monitoring? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install tensorboard
        echo -e "${GREEN}✓ Tensorboard installed.${NC}"
    fi
fi

# Check for deepspeed
if pip show deepspeed > /dev/null 2>&1; then
    echo -e "${GREEN}✓ DeepSpeed is installed.${NC}"
else
    echo -e "${YELLOW}⚠ DeepSpeed is not installed.${NC}"
    read -p "Would you like to install DeepSpeed for distributed training? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install deepspeed
        echo -e "${GREEN}✓ DeepSpeed installed.${NC}"
    fi
fi

# Setup cache manager
echo -e "\n${BLUE}Setting up cache manager...${NC}"
chmod +x cache_manager.py

echo -e "\n${GREEN}✅ TabulaPrima Dataset Cache installation complete!${NC}"
echo -e "${YELLOW}To get started, run: python train_cli.py${NC}"
echo -e "${YELLOW}To manage dataset caches, run: ./cache_manager.py list${NC}"