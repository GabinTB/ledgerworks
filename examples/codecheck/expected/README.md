# expected

The reference run, produced by the package author with
`codecheck-fomc all` followed by `codecheck-fomc freeze`: one directory per
artifact kind (`fomc_statements`, `fomc_embeddings`, `fomc_narrative_scores`),
each a verbatim copy of the artifact directory including its `meta.json`
(hyperparameters, inputs, model card, git commit, file digests), plus
`RUN.json` with the three artifact ids.

`codecheck-fomc compare` checks a fresh run against these files.
