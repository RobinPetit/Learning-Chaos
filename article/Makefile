SRC=article.tex
AUX=$(SRC:.tex=.aux)
BIB=article.bib

default: article.pdf

article.pdf: ${SRC} ${BIB}
	pdflatex ${SRC}
	bibtex ${AUX}
	pdflatex ${SRC}
	pdflatex ${SRC}

clean:
	rm -f article.pdf article.aux article.log article.log
