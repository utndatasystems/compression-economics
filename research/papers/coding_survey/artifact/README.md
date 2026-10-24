# Anonymous SIGMOD 2027 artifact source

This directory builds a standalone, review-stage repository for the main
experiments in the accompanying anonymous paper. The generated repository
contains the three reported CUDA operating points, the conventional and
external LLM baseline runners, and the two limits-of-token-parallelism
experiments. It deliberately excludes the manuscript, appendix ablations,
notebooks, exploratory programs, tests, cached models, datasets, and generated
archives.

Build the repository from the project root:

```bash
bash research/papers/coding_survey/artifact/build_anonymous_repository.sh \
  /tmp/sigmod-2027-artifact
```

The output path must not already exist. Review the generated tree and the
audit report printed by the script before putting it in a new private GitHub
repository. The generated `README.md` contains reviewer-facing setup and run
instructions.

This is an allowlist build: adding a file elsewhere in the working repository
does not silently add it to the review artifact. If a main experiment gains a
new runtime dependency, add that file explicitly to the allowlist in the build
script.
