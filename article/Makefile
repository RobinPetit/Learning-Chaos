SRC=article.tex
AUX=$(SRC:.tex=.aux)
BIB=$(SRC:.tex=.bib)
OUT=$(SRC:.tex=.pdf)

default: ${OUT}

article.pdf: ${SRC} ${BIB}
	pdflatex ${SRC}
	bibtex ${AUX}
	pdflatex ${SRC}
	pdflatex ${SRC}

clean:
	rm -f ${OUT} *.aux *.log *.log *.bbl *.blg *.out
