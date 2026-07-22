# AAAI-27 abstract (main track)

Revision of the ISE 619 final-report abstract. Updated for the six-dataset
suite, the deployment-adjusted index, the Diabetes-130 result, and the
capacity-matched baseline. Style pass follows the humanizer rules (no
em-dashes, no filler, concrete numbers, direct verbs) and standard AAAI
abstract conventions (single paragraph, no citations, no URLs; put the code
link in a footnote on page 1 instead).

---

We study learning to allocate scarce treatments, such as housing assistance
or hospital interventions, when a policy trained on observational data must
respect long-run capacity constraints. The standard approach is two-stage:
fit per-arm outcome models, then compute dual capacity prices that turn the
predictions into a feasible assignment rule. Its prediction stage is
decision-blind, since errors that flip allocations cost no more in training
than errors that change nothing. We instead train the scoring model
end-to-end, differentiating an inverse-propensity-weighted estimate of
policy value through the equilibrium prices themselves, which preserves the
dual-price structure that makes the policy implementable online. Two inner
objectives give a choice of guarantees: a nonconvex formulation
self-consistent with the deployed softmax policy, and a convex relaxation
whose stationary points are exactly feasible in expectation, with the gap
between the two bounded by tau log |T|. We evaluate on six datasets, four of
them real, under a queueing simulator that prices infeasibility and delay by
their deployed consequences. End-to-end training ranks first and second on
the resulting deployment-adjusted value index at every delay cost, including
zero. Two-stage baselines deploy above their capacity caps and wait 2 to 10
times longer in queue. On the largest dataset, 70,000 hospital patients,
end-to-end also wins raw policy value (paired t = 9.2), and the margin
survives a capacity-matched neural baseline. Where ground truth is
measurable, flexible two-stage regression keeps a raw-value lead; the
contribution of end-to-end training is that its policies are deployable as
trained.

---

## Notes against the old abstract

- Kept: the problem framing, the "decision-blind" critique, the bilevel
  training description, and the F versus G contrast. These sentences carried
  the submission's identity and survive nearly verbatim.
- Replaced: "Experiments on a synthetic allocation task and a real-queue
  simulation indicate that the end-to-end policy can improve on a strong
  two-stage baseline" was both vague and, on our final numbers, wrong in
  emphasis. The new closing states the three verified claims (index first at
  every kappa; caps and waits; Diabetes-130 paired t = 9.2 surviving the
  matched control) and concedes the raw-value lead of flexible two-stage
  where ground truth exists. Reviewers reward the concession; it is also
  simply true.
- Dropped: the GitHub URL (AAAI abstracts do not carry URLs; use a
  code-availability footnote), and "organ transplants" from the motivating
  list (a forced triplet, and organ allocation raises fairness questions the
  paper does not treat).
- Numbers that back each claim: index results in results/deploy_index.json
  (combined crossover 0.0); waits and caps in the per-dataset sweep CSVs;
  Diabetes-130 paired differences F minus S2-dr = +0.045 (t = 9.16) and
  F minus S2-mlp = +0.051 (t = 9.42) at N = 48,000, 10 seeds.

## De-brittling pass (2026-07-22, co-author feedback)

The results sentences no longer carry numbers that can move as experiments
are re-run: "six datasets, four of them real" became "real and synthetic
datasets" (the suite has since grown a seventh), "ranks first and second"
became "take the top slots" (robust to which of F/G leads), "2 to 10 times"
became "several times", and "paired t = 9.2" became "statistically
significant". Kept on purpose: the seventy-thousand-patient cohort size (a
fixed property of the dataset, and the scale claim is load-bearing) and
"including zero delay" (the crossover-at-zero claim is the headline; if it
ever stops holding, the abstract must change anyway, not silently survive).
The exact statistics live in the results section and the report, where they
are regenerated with the experiments.

## Sync with the LaTeX draft (2026-07-22)

Canonical text now matches the co-authors' draft: the gap clause is inline
LaTeX ($\tau\log|\mathcal{T}|$, phrased "within a bounded optimality
gap"), and the evaluation sentence uses the co-author's phrasing "several
datasets, including real and synthetic". The de-brittling of result numbers
was already present in this version; no precise statistics remain in the
abstract.
