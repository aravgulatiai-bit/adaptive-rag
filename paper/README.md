# Paper: Hierarchical Fuzzy Query-Complexity Routing for Cost-Efficient RAG

LaTeX source for the research paper describing this repository's system.

## Files

- `main.tex` — the full paper (single-column article, arXiv-preprint style).
- `references.bib` — bibliography (all real, verifiable references).

## Compile

No LaTeX toolchain is required locally — the easiest path is Overleaf:

1. Go to overleaf.com → New Project → Upload Project.
2. Zip this `paper/` folder and upload it (or create a blank project and add
   both files).
3. Set the compiler to pdfLaTeX (default) and compile. Two passes + BibTeX run
   automatically on Overleaf.

Locally (if you install TeX Live / MiKTeX):

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Where the numbers come from

Every value in the results tables is defined as a macro at the top of
`main.tex`, sourced from the repository's evaluation artifacts:

- SQuAD-200 experiment → `results_squad/eval_results.csv`
  (regenerate: `rag --config config/config_squad.yaml ingest` then
  `rag --config config/config_squad.yaml eval`)
- Enterprise-22 experiment → `results/eval_results.csv`
  (regenerate: `rag eval`)

If you re-run the evaluations, update the macro block in `main.tex` to match
the fresh CSVs before recompiling.

## Before submitting anywhere

- The author block currently lists the repository owner; adjust
  name/affiliation as needed.
- Venue formatting: the paper uses a generic `article` class. For a specific
  venue (ACL/NAACL, SIGIR, ICLR, etc.), port the content into that venue's
  official template and check page limits.
- The Limitations section honestly notes benchmark scale (200 + 22 questions,
  single machine, single run). For a top-tier venue, consider adding a larger
  benchmark (e.g., full HotpotQA subset), multiple runs for latency error
  bars, and a stronger reader model so quality differences can surface.
