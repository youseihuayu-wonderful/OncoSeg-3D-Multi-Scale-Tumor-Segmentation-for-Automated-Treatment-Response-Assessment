# Manager-Angle Interview Questions for OncoSeg

Questions a hiring manager, engineering manager, or tech lead would ask
about this project — focused on judgment, prioritization, risk,
collaboration, and business value rather than on implementation
details. Pair with `Interview_Questions.md` (IC/technical) and
`Project_QA_Log.md` (build-process) for full coverage.

---

## 1. Framing & motivation

1. In one sentence, what problem does OncoSeg solve and who pays to have it solved?
2. Why did you pick automated tumor segmentation instead of a simpler medical-AI problem (e.g. chest-X-ray classification)?
3. Why a hybrid CNN-Transformer instead of fine-tuning an existing model like nnU-Net or SwinUNETR?
4. What does success look like 6 months after this ships in a hospital? 18 months?
5. Who is the end user of the segmentation output — the radiologist, the oncologist, the trial coordinator, the billing system? How would the answer change the design?
6. If a regulator asked "why not just use RECIST as radiologists already do," what is your answer in 30 seconds?
7. What competing approaches exist today, and why is yours better or worse?

## 2. Technical decision-making & tradeoffs

8. Why cross-attention skip connections instead of standard concatenation? What is the theoretical and empirical justification?
9. You chose MC Dropout over deep ensembles — walk me through that tradeoff (compute, calibration, engineering cost).
10. Why 5 MC Dropout samples, not 20 or 50? How did you decide?
11. OncoSeg is 10.86M parameters vs SwinUNETR's 62M and UNETR's 130M. Is that a feature or a liability for the clinical use case?
12. ROI size of 128³ with 50% overlap — why those numbers, and what breaks if you change them?
13. You use sigmoid multi-label outputs (TC/WT/ET) instead of softmax over 4 classes. Defend that choice.
14. How did you decide on deep-supervision weights, and what would you change?
15. Training used AdamW with weight_decay=1e-5 — why those, and did you grid-search?
16. What is the weakest technical decision in the project, and why have you not fixed it?

## 3. Prioritization & scope

17. You have four open work items (SwinUNETR, UNETR, 4-variant ablation, LUMIERE evaluation). Rank them by business impact and defend the ranking.
18. What did you deliberately cut from scope, and what triggered that cut?
19. You added 24 new tests after the main training run was already done. Why invest in tests late rather than skipping them?
20. If you had 1 extra week, what would you add?
21. If you had to remove one contribution (cross-attention, uncertainty, RECIST integration), which goes first and why?
22. You have interview-prep docs in `docs/`. Should those live in a public research repo, or somewhere private? What is your reasoning?
23. You spent time on an MSD dataset integrity pre-flight checker. Justify the cost/value.

## 4. Validation & quality

24. How do you know the model actually works and isn't overfit to MSD?
25. 96 validation subjects is small. What are the statistical implications for your reported Dice gap?
26. Wilcoxon signed-rank on per-subject Dice gave p<0.01 on WT. What does that actually prove, and what does it not prove?
27. You report ECE = 0.0101. Is that measured on the same val set the model picked a checkpoint on? Is that a problem?
28. How would you validate this model before clinical deployment?
29. What is the minimum cohort size for a cross-institutional validation study you would trust?
30. If the model produces a confident but wrong segmentation, what does the system do?
31. Tests: 106 unit/integration passing. What is untested today that matters most?

## 5. Risk & failure modes

32. What is the most likely way this model causes patient harm in production?
33. What is the most likely way the project fails before reaching production?
34. Worst validation subject BRATS_077 had Dice 0.239 — what kind of tumor is this, and what does that failure pattern imply for deployment?
35. If a future MRI scanner version shifts the intensity distribution, what breaks and how do you find out?
36. Your longitudinal RECIST demo used morphologically-perturbed masks, not real second-timepoint scans. What is the actual risk of that validation gap?
37. What vendor lock-in does this project have? (MONAI, PyTorch versions, specific GPU architectures.)
38. Single-person project means a single point of failure. How would you de-risk that?

## 6. Team, collaboration, leadership

39. If you had three engineers for 6 months, how would you divide the work?
40. Which pieces of this project would you give to a junior engineer? Which to a senior?
41. Walk me through a code review you would do on a PR that adds a new loss function to this repo.
42. A team member argues we should replace cross-attention with plain concatenation to "simplify." Do you push back, escalate, or concede? How?
43. A radiologist partner says the uncertainty maps are "too noisy to use." What is your next conversation?
44. How would you onboard a new engineer to this codebase in their first week?
45. You have a disagreement with a data-science manager about whether Dice or HD95 is the right headline metric. How do you resolve it?

## 7. Timeline, resources, cost

46. How long did this project take end-to-end, and how long would a v2 take with what you now know?
47. What was the single biggest time-sink, and would you do it differently?
48. Compute cost so far: estimate it, and estimate what's remaining for the Kaggle run + LUMIERE eval.
49. You moved training from Colab to Kaggle after the 2026-04-06 Colab run failed. Talk me through that decision — what were the alternatives?
50. If you had a $5,000 compute budget, how would you spend it to maximize publishable results?
51. If you had $500 instead, what changes?
52. How much of the project was wall-clock time waiting for training vs active engineering work?

## 8. Productionization & scaling

53. Walk me from the current local-inference script to a deployed service that handles 1,000 scans per day.
54. How would you monitor this model in production?
55. What signals would trigger an automatic model-retraining cycle?
56. Where does HIPAA / PHI live in this pipeline, and what would you need to add for HIPAA compliance?
57. What is the inference latency budget for a clinical PACS workflow, and does OncoSeg fit?
58. How would you A/B test a new OncoSeg version against the current one in production without harming patient care?
59. If the marketing team promised "OncoSeg segments any tumor in 10 seconds," what is your response?

## 9. Clinical & regulatory

60. Which FDA pathway would this tool target (510(k), De Novo, PMA), and why?
61. What evidence package does the FDA require for this class of tool?
62. What is the difference between RECIST 1.1 and RANO, and why do both matter for a glioma response assessment tool?
63. A clinical trial sponsor wants to use OncoSeg as the primary endpoint-measurement tool. What additional validation would they demand?
64. If the model is biased — worse on certain subpopulations — how would you detect that, and how would you disclose it?
65. A radiologist overrides the model's segmentation. Where does that feedback go?

## 10. Retrospective & lessons

66. What mistake did you make in this project that cost you the most time?
67. What assumption did you make at the start that turned out to be wrong?
68. Which dependency or library decision would you reverse?
69. What did you learn about writing research code that your professors/coursework did not teach you?
70. If a hiring manager asked "is this a real engineering project or a side demo," what specific evidence do you point to?
71. What part of this project are you most proud of?
72. What part would you quietly rebuild if nobody was watching?

---

## How to use this list

- Do not memorize answers — the questions above have no single right answer. The interviewer is evaluating *how you reason*, not whether your decision matches theirs.
- For each question, prepare a 60-second core answer with one concrete supporting detail (a number, a commit, a failure story). Over-prepared fluent answers read as rehearsed.
- Expect follow-ups: "why not X instead," "what if the budget were halved," "walk me through a specific example." Have one real anecdote ready per category.
- When you don't know, say so and state what you would do to find out. Managers trust that more than confident fabrication.
