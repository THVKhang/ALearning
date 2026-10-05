"""Image batch -> token sequence, shared by the recurrent and Transformer models.

A 28x28 image has no natural notion of "time", so handbook Section 11.1 asks for an explicit
choice of sequence representation. Three are offered here and the choice is a config key, so
the same model can be trained on any of them:

* `rows`     - one timestep per image row, read top to bottom: 28 steps of 28 pixels.
* `columns`  - one timestep per image column, read left to right: 28 steps of 28 pixels.
* `patches`  - non-overlapping `patch_size` x `patch_size` blocks in row-major order:
               16 steps of 49 pixels at the default 7x7.

`rows` and `columns` keep one axis of the image intact inside each token and destroy the
other; `patches` keeps a square neighbourhood in each token, which is the representation a
Vision Transformer uses.
"""


MODES = ("rows", "columns", "patches")


def sequence_shape(input_shape, mode="rows", patch_size=7):
    """Return (timesteps, features_per_timestep) for `mode`, without building a tensor."""
    channels, height, width = input_shape
    if mode == "rows":
        return height, channels * width
    if mode == "columns":
        return width, channels * height
    if mode == "patches":
        if height % patch_size or width % patch_size:
            raise ValueError(f"patch_size {patch_size} does not divide a {height}x{width} "
                             "input evenly")
        return (height // patch_size) * (width // patch_size), channels * patch_size ** 2
    raise ValueError(f"unknown sequence mode: {mode} (expected one of {MODES})")


def to_sequence(x, mode="rows", patch_size=7):
    """(N, C, H, W) image batch -> (N, T, D) token sequence."""
    batch, channels, height, width = x.shape
    if mode == "rows":
        # (N, C, H, W) -> (N, H, C, W): timestep = row, features = that row of every channel.
        return x.permute(0, 2, 1, 3).reshape(batch, height, channels * width)
    if mode == "columns":
        return x.permute(0, 3, 1, 2).reshape(batch, width, channels * height)
    if mode == "patches":
        if height % patch_size or width % patch_size:
            raise ValueError(f"patch_size {patch_size} does not divide a {height}x{width} "
                             "input evenly")
        # Two unfolds cut the H and W axes into blocks: (N, C, H/p, W/p, p, p).
        blocks = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # Move the channel and in-patch axes last so each token is one flattened patch.
        blocks = blocks.permute(0, 2, 3, 1, 4, 5)
        return blocks.reshape(batch, -1, channels * patch_size ** 2)
    raise ValueError(f"unknown sequence mode: {mode} (expected one of {MODES})")
