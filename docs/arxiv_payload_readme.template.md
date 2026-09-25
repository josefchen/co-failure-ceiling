# arXiv submission payload, version 2 (September 2026)

**Paper:** Combining LLMs Rarely Beats the Single Best Model: A Provable Co-Failure Ceiling Across 67 Frontier Models
**Author:** Josef Chen (KAIKAKU) · arXiv:2606.27288 (replacement)

```
arxiv_payload/
  main.pdf                  the compiled paper
  arxiv-source.tar.gz       >>> upload this to arXiv (LaTeX source, pdflatex, .bbl included) <<<
  arxiv/                    the same source, unpacked
  PAYLOAD_README.md         this file
```

Code and one-command reproduction: https://github.com/josefchen/co-failure-ceiling
Data (pinned revision `{{HfRevision}}`): https://huggingface.co/datasets/josefchen/co-failure-67-models

## Submitting the replacement

1. On arxiv.org, open 2606.27288 and choose *Replace*. Upload `arxiv-source.tar.gz`.
2. Paste the title and abstract below. Comments field: "v2: corrects the first version's co-failure measurements; see
   Appendix (Revision history and data audit)".
3. The arXiv listing title currently differs from the PDF title; the replacement form is where to align them.

## Title

Combining LLMs Rarely Beats the Single Best Model: A Provable Co-Failure Ceiling Across 67 Frontier Models

## Abstract (plain text for the arXiv form; generated from paper/main.tex)

{{PaperAbstract}}
