# Coursework Rules And Compliance Notes

## Binding Rules From The Q&A

1. We are expected to implement all baselines across the three required frameworks, not just the random baseline.
2. We should reproduce the paper results as faithfully as possible.
3. We must not use implementation code from the authors or other external implementations.
4. We may reuse standard components that existed before the paper, for example an untrained `ResNet-18`, provided we train and integrate them ourselves.
5. The improvement section may modify one or several aspects of the algorithm, as long as the report explains the rationale and analyses the outcome.

## Repo Implications

- The TPCRP logic must remain authored in this repository.
- Third-party libraries are acceptable only as generic building blocks.
- Baseline runs and improvement runs should be stored separately.
- The report must distinguish clearly between:
  - faithful baseline reproduction;
  - the required baseline comparisons across frameworks;
  - the modified improvement and its justification.

## Current Compliance Check

- Custom TPCRP-style selection implementation: in place.
- Standard backbone reuse without external author code: in place.
- Separate folders for experiments, package code, results, and report: in place.
- Shared SimCLR checkpoint and embedding cache for reproducible GPU runs: in place.
- Framework-specific runners for supervised, embedding, and semi-supervised experiments: in place.
- Drafted improvement rationale: in place.
- All required baselines across the three frameworks: not complete yet.
- Strict FlexMatch-equivalent semi-supervised reproduction: not complete yet.

## Algorithm Implementation Marks

For the implementation-heavy part of the coursework, the current repo already covers the main technical ingredients of the paper method:

- self-supervised representation learning on the unlabeled pool;
- embedding extraction;
- clustering for diversity;
- typicality scoring from nearest-neighbour distances;
- selection from underrepresented clusters;
- iterative query-and-train evaluation.

This is a strong basis for the algorithm implementation marks. The larger remaining risk is not "the algorithm is missing", but "the comparison matrix and evidence are still incomplete".

## Practical Guardrails For Further Work

- Do not paste in code from paper repositories or blog posts.
- Keep every new experiment in its own result subfolder.
- Record exact flags used for every baseline and every modified run.
- In the report, be explicit when a result is only a smoke test or debug run.
