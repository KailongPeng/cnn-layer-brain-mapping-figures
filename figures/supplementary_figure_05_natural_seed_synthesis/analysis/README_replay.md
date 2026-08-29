# Figure 1D seed-controlled replay

This directory keeps the historical Figure 1D script unchanged and provides a
separate replay entry point with explicit input and no-overwrite gates.

The replay intentionally uses the assets at their historical Milgram locations:

`/gpfs/milgram/project/turk-browne/projects/localize/utils`

`/gpfs/milgram/project/turk-browne/users/kp578/localize/utils/tensorflow_inception_graph.pb`

The replay accepts one required scientific choice: `noise` uses the historical
random initialization, whereas `nsd` uses the 16 publication-compatible NSD
images listed in `ImageSynthesis/seed_manifests/r2q1_structured_seed_16.txt`.
The `ccby4` mode uses the separately tracked 16-image Wikimedia Commons set in
`ImageSynthesis/seed_manifests/cc_by_4_0_natural_seed_16.txt`. These are natural
images, not NSD stimuli. Their source pages and licenses are recorded in
`NSD_similarity/PLOS_cc_by_4_0_seed_candidate_attribution_16.csv`.
Image seeds are decoded as RGB and resized to the historical starting resolution
of 196 x 196. Both modes consume the same initial random draws, keeping all
subsequent random-crop draws aligned.

The first run should be a one-magnification, one-iteration diagnostic. It is not
a completed Figure 1D synthesis:

On Milgram:

```bash
cd /gpfs/milgram/project/turk-browne/projects/localize_paper
git pull
bash fig1d/submit_fig1d_noise_replay_pilot.sh noise
bash fig1d/submit_fig1d_noise_replay_pilot.sh nsd
bash fig1d/submit_fig1d_noise_replay_pilot.sh ccby4
```

The submission wrapper refuses a dirty worktree, checks every historical asset,
uses a unique tag containing the commit and UTC time, and writes only under:

`/gpfs/milgram/scratch60/turk-browne/kp578/localize/fig1d_replay`

It also submits an `afterany` postflight job. The postflight fails unless the
parent job completed and all expected pilot products pass validation.

During a run, each completed iteration prints a `PROGRESS` line with the recent
seconds per iteration, remaining time, and estimated UTC finish. The same state
is written atomically to `progress.json` in the run directory. To inspect it:

```bash
python3 -m json.tool /path/to/run/progress.json
```

The ETA uses the mean duration of up to the latest 10 iterations. Costs of
future magnifications are scaled by image area, so the estimate is provisional
after the first iteration and stabilizes as the run proceeds.

The synthesis writes a checkpoint after iteration 0, every 50 iterations, and
the final iteration of each magnification. Generate a 4 x 4 preview from the
latest available checkpoint with the exact scratch run directory printed by the
submission command:

```bash
python3 fig1d/viz_replay_checkpoint.py --run-dir /full/scratch/run/directory
```

To inspect a particular checkpoint:

```bash
python3 fig1d/viz_replay_checkpoint.py \
  --run-dir /full/scratch/run/directory --mag 0 --iter 50
```

To inspect the provisional Sfig5C matrix at a checkpoint, use the checkpoint's
stored RSM history:

```bash
source env_setup_glmsingle.sh
python3 -u fig1d/analyze_sfig5c_checkpoint.py \
  --run-dir /full/scratch/run/directory \
  --mag 3 --iter 100
```

This writes a PDF, PNG, NPZ, and text summary under `fig1d/_preview_transfer/`.
The outputs are deliberately labelled diagnostic. The historical replay saves
`Vars.pkl` before appending the current iteration's RSM, so a `currIter100`
checkpoint contains post-update iteration-100 images but an RSM history ending
at iteration 99. This is adequate for an interim convergence check; the final
panel should be recomputed from the selected final images.

After completion, inspect the job state and the final validation marker:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS,ExitCode
tail -n 80 logs/fig1d_MODE_replay_JOB_ID.out
tail -n 80 logs/fig1d_MODE_replay_JOB_ID.err
```

Success requires `COMPLETED`, exit code `0:0`, and the corresponding mode in the
job marker (`noise` or `nsd`):

```text
FIG1D REPLAY VALIDATION PASSED
FIG1D noise REPLAY PILOT PASSED
```

Do not use this pilot output as a final figure. A full historical replay and the
NSD-seeded variant should only be enabled after this baseline gate passes.

After the pilot passes, submit the complete historical run as one `psych_week`
job. This uses four magnifications and the original iteration counts
(`400/400/400/600`):

```bash
bash fig1d/submit_fig1d_noise_replay_full.sh noise
bash fig1d/submit_fig1d_noise_replay_full.sh nsd
bash fig1d/submit_fig1d_noise_replay_full.sh ccby4
```

For the replacement run, submit the `ccby4` pilot first. Proceed to the full
job only after the pilot job and its automatic postflight both report
`COMPLETED`/`PASSED`. The full submission writes to a unique timestamped
`fig1d_replay_ccby4_full_*` directory, prints a `progress.json` path, and never
resumes or overwrites the previous NSD-seeded run.

The full-run success markers are:

```text
FIG1D REPLAY VALIDATION PASSED
FIG1D noise FULL REPLAY PASSED
FIG1D REPLAY AUTOMATIC POSTFLIGHT PASSED
```

## Final image-seeded Sfig5/Sfig17 analysis

After the complete NSD run passes its automatic postflight, recompute the RSMs
from the final post-update PNGs. Do not substitute the last checkpoint
`Vars.pkl`: it was saved before the final iteration's RSM was appended.

```bash
bash fig1d/submit_sfig5_final_analysis.sh \
  /gpfs/milgram/scratch60/turk-browne/kp578/localize/fig1d_replay/fig1d_replay_nsd_full_c3aa8a8_20260820T181629Z
```

The job checks that the source is a completed four-magnification NSD run and
that the 16 stage-level final PNGs are byte-identical to `currIter599`. It then
runs the 12 CNN layers on those final images, builds the 12 image RSMs and their
120-element upper-triangle fingerprints, and saves the cross-layer correlation
matrix together with compact Panel B images in:

`rebuttal/analyses/structured_seed_data/nsd_seed_final_synthesis_exact.npz`

Success requires both markers:

```text
SFIG5 FINAL EXACT ANALYSIS PASSED
SFIG5 FINAL ANALYSIS AUTOMATIC POSTFLIGHT PASSED
```

For the completed CC BY 4.0-seeded run, select the corresponding manifest,
license record, and a separate no-overwrite output cache with the second
argument:

```bash
bash fig1d/submit_sfig5_final_analysis.sh \
  /gpfs/milgram/scratch60/turk-browne/kp578/localize/fig1d_replay/fig1d_replay_ccby4_full_34c625f_20260827T133041Z \
  ccby4
```

This writes
`rebuttal/analyses/structured_seed_data/ccby4_seed_final_synthesis_exact.npz`.
The completed two-day synthesis directory is read only; neither this analysis
nor its postflight modifies or resumes it.

Commit and push that NPZ, pull it on the local workstation, then render and
audit the one-page vector/raster PDF locally:

```bash
python3 -u rebuttal/analyses/compose_structured_seed_supplement.py
```

The staged output retains the historical upload filename
`figures/final/Sfig5_Structured_seed_image_synthesis.pdf`.
Panel A is the tracked 16-image seed manifest, Panel B is the corresponding 16
final synthesized images, and Panel C is recomputed from exactly those Panel B
images.
