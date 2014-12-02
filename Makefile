# Simple makefile to quickly access handy build commands for Cython extension
# code generation.  Note that the actual code to produce the extension lives in
# the setup.py file, this Makefile is just meant as a command
# convenience/reminder while doing development.

PYTHON ?= python
PKGDIR=brainsearch

help:
	@echo "Numpy/Cython tasks.  Available tasks:"
	@echo "ext  -> build the Cython extension module."
	@echo "cython-html -> create annotated HTML from the .pyx sources"
	@echo "test -> run a simple test demo."
	@echo "all  -> Call ext, html and finally test."

all: ext cython-html test

ext: imagespeed.so

test: ext
	nosetests .

cython-html:  ${PKGDIR}/imagespeed.html

imagespeed.so: ${PKGDIR}/imagespeed.pyx

	$(PYTHON) setup.py build_ext --inplace

# Phony targets for cleanup and similar uses

.PHONY: clean

clean:
	- find ${PKGDIR} -name "*.so" -print0 | xargs -0 rm
	- find ${PKGDIR} -name "*.pyd" -print0 | xargs -0 rm
	- find ${PKGDIR} -name "*.c" -print0 | xargs -0 rm
	- find ${PKGDIR} -name "*.html" -print0 | xargs -0 rm
	rm -rf build

distclean: clean
	rm -rf dist

# Suffix rules
%.c : %.pyx
	cython $<

%.html : %.pyx
	cython -a $<
