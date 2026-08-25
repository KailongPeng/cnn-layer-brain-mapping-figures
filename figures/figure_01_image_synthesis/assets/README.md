# Synthesis assets

`selected_channels_16.json` is a readable record of the channels used for the
16-image experiment. The exact historical run requires four binary assets:

- `16corr_GradOrth.npy` is the archived `12 x 16 x 16` target matrix.
- `bestchans16.pkl` contains the selected channels in the format read by the
  historical scripts.
- `corrsvals16.pkl` contains the corresponding channel-selection summary.
- `tensorflow_inception_graph.pb` is the pretrained TensorFlow Inception graph.
  It is downloaded separately and ignored by Git.

The three small experiment-specific assets are included in this directory and
are also archived on Milgram at:

```text
/gpfs/milgram/project/turk-browne/projects/localize/utils/16corr_GradOrth.npy
/gpfs/milgram/project/turk-browne/projects/localize/utils/bestchans16.pkl
/gpfs/milgram/project/turk-browne/projects/localize/utils/corrsvals16.pkl
```

Expected SHA-256 checksums are:

```text
a9ca82699e7f2ac95c94226706978f835db5ec0cb99b9383168c598ce699f35a  16corr_GradOrth.npy
0f66a183b39a104c75832299c718442990d5530832e04d90e3def8ffdfb6f89d  bestchans16.pkl
fddad4814e1ec6de39d8876ccbbe23068c9260d4c35fb3b7e764c59b93adbb43  corrsvals16.pkl
```

The target matrix must have shape `12 x 16 x 16`. The model graph is available
in Google's `inception5h.zip` archive:

```bash
curl -LO https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
unzip inception5h.zip tensorflow_inception_graph.pb
mv tensorflow_inception_graph.pb figures/figure_01_image_synthesis/assets/
```

Do not substitute newly generated assets when reproducing the reported
experiment. Verify the included files against the checksums above. The model
graph is intentionally not tracked because it is about 54 MB and can be
obtained from its upstream archive.
