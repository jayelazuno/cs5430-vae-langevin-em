#!/bin/bash

set -e

cd reports

pandoc final_report.md \
  --standalone \
  --toc \
  --mathjax \
  --css pandoc_report.css \
  -o final_report.html

pandoc final_report.md \
  --standalone \
  --toc \
  --pdf-engine=tectonic \
  -o final_report.pdf
