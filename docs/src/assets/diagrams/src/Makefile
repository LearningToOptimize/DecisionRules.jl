# Builds the TS-DDR diagram (a redraw of the published TS-DDR figure,
# arXiv:2405.14973) from tsddr.tex: PDF with pdflatex, SVG with pdftocairo,
# then a white rectangle under the drawing so it reads on dark pages too.
#
#   make -C docs/src/assets/diagrams/src
#
# Needs a TeX Live with pdflatex and poppler's pdftocairo.
OUT := ..
TMP := build

all: $(OUT)/tsddr.svg $(OUT)/tsddr.pdf $(OUT)/tsddr-compact.svg

$(TMP)/%.pdf: tsddr.tex
	mkdir -p $(TMP)
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=$(TMP) -jobname=$* tsddr.tex >/dev/null

$(OUT)/tsddr.pdf: $(TMP)/tsddr.pdf
	cp $< $@

$(OUT)/%.svg: $(TMP)/%.pdf
	pdftocairo -svg $< $(TMP)/$*.raw.svg
	perl -0pe 's/(<svg [^>]*>)/$$1\n<rect width="100%" height="100%" fill="#ffffff"\/>/' $(TMP)/$*.raw.svg > $@

clean:
	rm -rf $(TMP)

.PHONY: all clean
