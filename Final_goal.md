# darkvessel — final goal

## The goal

Build an open, reproducible pipeline that finds **dark vessels** — ships visible in Sentinel-1
radar but not broadcasting on AIS — over a real study area and time period, and that reports
where they concentrate, with every claim traceable back to the evidence behind it.

The finished project answers three questions:

1. **Which radar detections have no AIS declaration to explain them?** Per scene, per
   detection, with the reason stated (matched to what, at what distance, interpolated or
   nearest, how much the azimuth correction moved it).
2. **Where and when do dark vessels concentrate?** Across the whole archive, related to
   distance from shore, depth, fishing effort and EEZ boundaries.
3. **How far can those results be trusted?** Detector precision/recall, what the AIS search
   actually covered, and which results are only candidates for inspection.

## Deliverables

- [x] **Tier 1 core** — tiling, pixel→ground, AIS interpolation, azimuth correction, optimal
      matching, structure register, recurrence clustering, CLI. Runs with no network.
- [x] **Interactive viewer** — live and static, showing why each detection got its status.
- [ ] **Real data run** — at least one real Sentinel-1 scene fused with real DMA AIS,
      producing a sensible result.
- [ ] **Trained detector** — a CNN trained on LS-SSDD, satisfying the `Detector` protocol,
      with reported precision/recall and compared against a simple CFAR baseline.
- [ ] **Tier 2 contextual layers** — distance to shore, depth, fishing effort, EEZ from
      Earth Engine, joined to every detection.
- [ ] **Contrastive embeddings** — detection crops embedded for structure/vessel separation
      and similarity search.
- [ ] **Archive-wide run** — every scene in the study period, processed from a single config.
- [ ] **Concentration analysis and static map** — where dark detections cluster, normalised
      for how often each area was imaged.
- [ ] **Write-up** — README/report covering method, the design decisions, results and their limits.
- [ ] **Published** — code on GitHub with CI green; the viewer on GitHub Pages.

## Milestones

| # | Milestone | Done when |
|---|---|---|
| 1 | Core + viewer | ✅ 150 tests pass, viewer works in browser |
| 2 | Online | Repo pushed, CI green, Pages bundle live |
| 3 | First real scene | One Sentinel-1 scene + DMA AIS runs end to end; results inspected by eye in the viewer |
| 4 | Real detector | Trained model beats the CFAR baseline on LS-SSDD test split; plugged into the pipeline |
| 5 | Context | Each detection carries its contextual variables; missing values are absent, not zero |
| 6 | Archive | Full study period processed reproducibly |
| 7 | Analysis | Concentration map and statistics, corrected for revisit frequency |
| 8 | Write-up | Report finished, results stated with their uncertainty |

## Definition of done

- Anyone can clone the repo, run the quick start with no credentials, and get the README's
  numbers.
- Anyone with the listed accounts can reproduce the real-data results from the configs.
- Every `dark` detection can be opened in the viewer and its status explained.
- Detector performance and the AIS coverage behind each `dark` claim are reported alongside
  the findings.

## Out of scope

- Identifying specific vessels or owners, or claiming wrongdoing. A dark detection is a claim
  about the evidence searched, not a verdict.
- Real-time or operational monitoring.
- SAR sensors other than Sentinel-1.

## Still to decide

- Exact study area (the default CRS, EPSG:25832, points at Danish waters, which matches the DMA
  AIS source) and time period.
- Whether the target is a paper, a portfolio project, or a tool for others — this changes how
  much the write-up and hosting matter.
