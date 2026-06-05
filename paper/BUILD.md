# Building paper.pdf

## Requirements

A working TeX distribution with `pdflatex` and `bibtex`. On macOS, the
easiest path is MacTeX (full) or BasicTeX + packages:

```bash
# MacTeX (recommended, ~5 GB)
brew install --cask mactex-no-gui

# BasicTeX (minimal, ~100 MB; then add packages as needed)
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install microtype booktabs caption natbib hyperref \
                   geometry amsmath amssymb graphicx xcolor
```

On Debian/Ubuntu:

```bash
sudo apt install texlive-latex-recommended texlive-bibtex-extra \
                 texlive-latex-extra texlive-fonts-recommended
```

## Build

From the `paper/` directory, run the standard 4-pass sequence:

```bash
cd paper
pdflatex paper.tex          # 1st pass — generates .aux with citation keys
bibtex   paper              # resolves citations against references.bib
pdflatex paper.tex          # 2nd pass — pulls in .bbl, leaves refs unresolved
pdflatex paper.tex          # 3rd pass — resolves all cross-references
```

Or in one line:

```bash
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

Output: `paper.pdf` in the current directory.

## Latexmk shortcut (if installed)

```bash
latexmk -pdf paper.tex
```

`latexmk` handles the pass-count automatically and stops when the
auxiliary files are stable.

## Inputs

- `paper.tex`         — main source
- `references.bib`    — 15 entries, all cited
- `figures/fig{1..5}_*.pdf` — vector figures, included via
  `\includegraphics`; the `\graphicspath{{figures/}}` directive lets
  the `.tex` reference them by filename only
- `figures/scripts/`  — reproducible generation scripts (not required
  for build; only for regenerating figures from `results/*.csv`)

## Reproducing figures before build

If `figures/*.pdf` are missing or out of date, regenerate them from the
repo root with the project venv:

```bash
source .venv/bin/activate
for f in paper/figures/scripts/fig*.py; do python "$f"; done
```

See `paper/figures/README.md` for details.

## Clean

```bash
rm -f paper.{aux,bbl,blg,log,out,toc}
```
