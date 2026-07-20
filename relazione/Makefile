# Makefile for Relazione LaTeX document

# Main document name (without .tex extension)
MAIN = relazione

# Output directory
OUTDIR = out

# LaTeX compiler
LATEX = pdflatex
LATEXMK = latexmk

# LaTeX compiler flags
LATEXFLAGS = -output-directory=$(OUTDIR) -interaction=nonstopmode
LATEXMKFLAGS = -pdf -output-directory=$(OUTDIR) -pdflatex="pdflatex -interaction=nonstopmode"

# Default target
all: $(OUTDIR)/$(MAIN).pdf

# Create output directory if it doesn't exist
$(OUTDIR):
	mkdir -p $(OUTDIR)

# Compile the main document using latexmk
$(OUTDIR)/$(MAIN).pdf: $(MAIN).tex homeworks/*/*.tex project/*.tex | $(OUTDIR)
	$(LATEXMK) $(LATEXMKFLAGS) $(MAIN).tex

# Continuous preview compilation
preview: $(OUTDIR)/$(MAIN).pdf | $(OUTDIR)
	zathura $(OUTDIR)/$(MAIN).pdf & \
	$(LATEXMK) -pdf -pvc -view=none -output-directory=$(OUTDIR) -pdflatex="pdflatex -interaction=nonstopmode" $(MAIN).tex

# Quick compile (single pass, no bibliography) - fallback to pdflatex
quick: | $(OUTDIR)
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# Clean output directory
clean:
	$(LATEXMK) -c -output-directory=$(OUTDIR)
	rm -rf $(OUTDIR)/*

# Clean and rebuild
rebuild: clean all

# View the PDF (Linux)
view: $(OUTDIR)/$(MAIN).pdf
	zathura $(OUTDIR)/$(MAIN).pdf

# Help target
help:
	@echo "Available targets:"
	@echo "  all     - Full compilation using latexmk"
	@echo "  preview - Continuous preview compilation with auto-refresh"
	@echo "  quick   - Quick compilation (single pass, no bibliography)"
	@echo "  clean   - Remove all output files"
	@echo "  rebuild - Clean and full rebuild"
	@echo "  view    - Open the compiled PDF"
	@echo "  help    - Show this help message"

.PHONY: all preview quick clean rebuild view help
