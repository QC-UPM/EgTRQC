SPHINXBUILD ?= python -m sphinx
SOURCEDIR = source
BUILDDIR = _build

.PHONY: html clean

html:
	PYTHONPATH=../src $(SPHINXBUILD) -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
