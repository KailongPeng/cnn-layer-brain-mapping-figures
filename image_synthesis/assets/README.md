# Synthesis assets

`selected_channels_16.json` records the channels used for the 16-image experiment.
Two more files are required to run the synthesis script:

- `16corr_GradOrth.npy` is the archived target matrix used in the experiment. It
  is small and should be tracked in this directory.
- `tensorflow_inception_graph.pb` is the pretrained TensorFlow Inception graph.
  It is downloaded separately and ignored by Git.

The target matrix must have shape `12 x 16 x 16`. The model graph is available
in Google's `inception5h.zip` archive:

```bash
curl -LO https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
unzip inception5h.zip tensorflow_inception_graph.pb
mv tensorflow_inception_graph.pb image_synthesis/assets/
```

Do not substitute a newly generated target matrix when reproducing the reported
experiment. Use the archived `16corr_GradOrth.npy` from the original run.
