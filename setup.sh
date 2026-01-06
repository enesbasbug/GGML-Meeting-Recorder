#!/bin/bash
#
# GGML Meeting Recorder - Setup Script
# =====================================
# This script sets up everything you need to run the Meeting Recorder.
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       GGML Meeting Recorder - Setup Script                ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for required tools
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is required but not installed.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python3 found${NC}"

if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: cmake is required. Install with: brew install cmake${NC}"
    exit 1
fi
echo -e "${GREEN}✓ CMake found${NC}"

if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is required.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Git found${NC}"

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo -e "${YELLOW}Warning: This is optimized for Apple Silicon (M1/M2/M3/M4).${NC}"
    echo -e "${YELLOW}Performance on Intel Macs may vary.${NC}"
fi

echo ""
echo -e "${YELLOW}Step 1: Installing Python dependencies...${NC}"
pip3 install rumps sounddevice numpy scipy huggingface-hub --user --quiet
echo -e "${GREEN}✓ Python dependencies installed${NC}"

echo ""
echo -e "${YELLOW}Step 2: Setting up whisper.cpp...${NC}"
if [ ! -d "whisper.cpp" ]; then
    echo "Cloning whisper.cpp..."
    git clone https://github.com/ggml-org/whisper.cpp.git
fi

cd whisper.cpp
if [ ! -f "build/bin/whisper-cli" ]; then
    echo "Building whisper.cpp with Metal support..."
    mkdir -p build && cd build
    cmake .. -DGGML_METAL=ON
    cmake --build . --config Release -j 8
    cd ..
fi
echo -e "${GREEN}✓ whisper.cpp built${NC}"

# Download Whisper models
echo ""
echo -e "${YELLOW}Step 3: Downloading Whisper models...${NC}"
if [ ! -f "models/ggml-base.en.bin" ]; then
    echo "Downloading Whisper base.en model (141MB)..."
    ./models/download-ggml-model.sh base.en
fi
echo -e "${GREEN}✓ Whisper base.en model ready${NC}"

if [ ! -f "models/ggml-large-v3.bin" ]; then
    echo "Downloading Whisper large-v3 model (2.9GB)..."
    echo "(This may take a few minutes...)"
    ./models/download-ggml-model.sh large-v3
fi
echo -e "${GREEN}✓ Whisper large-v3 model ready${NC}"

cd "$SCRIPT_DIR"

echo ""
echo -e "${YELLOW}Step 4: Setting up llama.cpp...${NC}"
if [ ! -d "../llama.cpp" ] && [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp.git
    cd llama.cpp
    mkdir -p build && cd build
    cmake .. -DGGML_METAL=ON
    cmake --build . --config Release -j 8
    cd "$SCRIPT_DIR"
    LLAMA_PATH="$SCRIPT_DIR/llama.cpp"
else
    if [ -d "../llama.cpp" ]; then
        LLAMA_PATH="$(cd ../llama.cpp && pwd)"
    else
        LLAMA_PATH="$SCRIPT_DIR/llama.cpp"
    fi
fi
echo -e "${GREEN}✓ llama.cpp ready at: $LLAMA_PATH${NC}"

echo ""
echo -e "${YELLOW}Step 5: Downloading LLM models...${NC}"
mkdir -p models

# DeepSeek R1 8B
if [ ! -f "models/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf" ]; then
    echo "Downloading DeepSeek R1 8B (4.6GB)..."
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF', 'DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf', local_dir='models')"
fi
echo -e "${GREEN}✓ DeepSeek R1 8B ready${NC}"

# Qwen 2.5 7B
if [ ! -f "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf" ]; then
    echo "Downloading Qwen 2.5 7B (4.4GB)..."
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('bartowski/Qwen2.5-7B-Instruct-GGUF', 'Qwen2.5-7B-Instruct-Q4_K_M.gguf', local_dir='models')"
fi
echo -e "${GREEN}✓ Qwen 2.5 7B ready${NC}"

# Llama 3.2 3B (small, fast)
if [ ! -f "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf" ]; then
    echo "Downloading Llama 3.2 3B (1.9GB)..."
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('bartowski/Llama-3.2-3B-Instruct-GGUF', 'Llama-3.2-3B-Instruct-Q4_K_M.gguf', local_dir='models')"
fi
echo -e "${GREEN}✓ Llama 3.2 3B ready${NC}"

echo ""
echo -e "${YELLOW}Step 6: Creating directories...${NC}"
mkdir -p recordings
mkdir -p output
echo -e "${GREEN}✓ Directories created${NC}"

echo ""
echo -e "${YELLOW}Step 7: Making scripts executable...${NC}"
chmod +x MeetingRecorder.py 2>/dev/null || true
chmod +x Start.command 2>/dev/null || true
chmod +x setup.sh 2>/dev/null || true
echo -e "${GREEN}✓ Scripts are executable${NC}"

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Setup Complete! 🎉                      ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "To start the Meeting Recorder:"
echo -e "  ${BLUE}python3 MeetingRecorder.py${NC}"
echo ""
echo -e "Or double-click: ${BLUE}Start.command${NC}"
echo ""
echo -e "Look for the ${BLUE}🎙️ icon${NC} in your menu bar!"
echo ""
echo -e "${YELLOW}Models installed:${NC}"
ls -lh models/*.gguf 2>/dev/null | awk '{print "  - " $NF " (" $5 ")"}'
echo ""
echo -e "${YELLOW}Optional: Download premium models for better quality:${NC}"
echo -e "  See README.md for Llama 3.3 70B, DeepSeek R1 32B, Qwen 2.5 72B"
echo ""

