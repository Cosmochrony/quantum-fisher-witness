#!/bin/bash

# Quantum Fisher Information witness paper compilation script.
# Compiles quantum_fisher_witness.tex with bibliography support.
# Figures live in tex/figures/ and are resolved via \graphicspath; the script
# therefore runs pdflatex from tex/ and writes all artefacts to out/.

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
MAIN_NAME="quantum_fisher_witness"
OUTPUT_DIR="$ROOT/out"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
echo -e "${GREEN}=== Compiling Quantum Fisher witness paper ===${NC}"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT/tex"

echo -e "${GREEN}Step 1/4: First pdflatex run${NC}"
pdflatex -file-line-error -interaction=nonstopmode -synctex=1 \
    -output-directory="$OUTPUT_DIR" "$MAIN_NAME.tex"

echo -e "${GREEN}Step 2/4: Running bibtex${NC}"
( cd "$OUTPUT_DIR" && BIBINPUTS="$ROOT/tex:" bibtex "$MAIN_NAME" )

echo -e "${GREEN}Step 3/4: Second pdflatex run${NC}"
pdflatex -file-line-error -interaction=nonstopmode -synctex=1 \
    -output-directory="$OUTPUT_DIR" "$MAIN_NAME.tex"

echo -e "${GREEN}Step 4/4: Third pdflatex run${NC}"
pdflatex -file-line-error -interaction=nonstopmode -synctex=1 \
    -output-directory="$OUTPUT_DIR" "$MAIN_NAME.tex"

if [ -f "$OUTPUT_DIR/$MAIN_NAME.pdf" ]; then
    echo -e "${GREEN}=== Compilation successful ===${NC}"
    ls -lh "$OUTPUT_DIR/$MAIN_NAME.pdf"
else
    echo -e "${RED}=== Compilation failed ===${NC}"
    echo -e "Check $OUTPUT_DIR/$MAIN_NAME.log for errors"
    exit 1
fi
