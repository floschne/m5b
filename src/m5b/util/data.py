import re
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import webdataset as wds


def is_wds_url(wds_path: str | Path) -> bool:
    return len(re.findall(r".*\{\d+..\d+\}.tar", str(wds_path))) > 0


def generate_wds_url_from_wds_path(wds_path: str | Path) -> str:
    wds_path = Path(wds_path)
    shard_fns = list(wds_path.glob("*.tar"))

    def _extract_shard_number(shard_fn: Path) -> Tuple[str, str]:
        stem = shard_fn.stem
        shard_num = re.findall(r"0\d+$", stem)
        assert len(shard_num) == 1, f"Could not extract shard number from {stem}"
        shard_stem = re.sub(f"{shard_num[0]}$", "", stem)
        return shard_num[0], shard_stem

    shard_nums, shard_stems = zip(*sorted(map(_extract_shard_number, shard_fns)))
    assert (
        len(set(shard_stems)) == 1
    ), f"Found multiple shard stems in {wds_path}: {shard_stems}"

    return f"{wds_path}/{shard_stems[0]}{{{shard_nums[0]}..{shard_nums[-1]}}}.tar"


def build_wds(
    wds_path: str | Path,
    ds_size: int | None = None,
    decode: str = "pil",
    tuple_content: Tuple[str, ...] = ("jpg;png", "json"),
    map_tuple: Tuple[Callable, ...] | None = None,
    batch_size: int | None = None,
    shuffle: bool = True,
    shuffle_buffer: int = 1000,
) -> wds.WebDataset:
    if not is_wds_url(wds_path):
        wds_path = generate_wds_url_from_wds_path(wds_path)
    # see https://github.com/webdataset/webdataset#webdataset
    ds = wds.WebDataset(urls=str(wds_path))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer)
    ds = ds.decode(decode).to_tuple(*tuple_content)
    if map_tuple is not None:
        ds = ds.map_tuple(*map_tuple)
    if ds_size is not None:
        # FIXME: this is not working with more than one worker
        #  (i.e., in combination with a dataloader)
        # bs = batch_size if batch_size else 1
        # ds = ds.with_epoch(int(ds_size / bs)).with_length(ds_size)
        ds = ds.with_length(ds_size)
    if batch_size is not None:
        ds = ds.batched(batch_size)
    return ds


def build_wds_dataloader(
    ds: wds.WebDataset,
    collate_fn: Callable | None = None,
    num_workers: int = 1,
    pin_memory: bool = True,
) -> wds.WebLoader:
    # see https://github.com/webdataset/webdataset#dataloader
    loader = wds.WebLoader(
        ds,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=None,
    )
    return loader


def create_splits_from_df(
    df,
    splits: Dict[str, float] = {"train": 0.7, "test": 0.2, "val": 0.1},
    random_state: int = 42,
) -> Dict[str, pd.DataFrame]:
    dfs = dict()
    assert np.isclose(
        np.sum(list(splits.values())), 1.0
    ), f"Split sizes must sum to 1, got sum({splits}) = {np.sum(list(splits.values()))}"
    splits = {
        k: int(v * len(df)) for k, v in sorted(splits.items(), key=lambda item: item[1])
    }
    split_sizes = list(splits.values())
    shuffled = df.sample(frac=1.0, random_state=random_state)

    for i, split_name in enumerate(splits.keys()):
        if i == 0:
            dfs[split_name] = shuffled[: split_sizes[i]]
        else:
            dfs[split_name] = shuffled[
                split_sizes[i - 1] : split_sizes[i - 1] + split_sizes[i]
            ]

        print(
            f"{split_name}: Relative {len(dfs[split_name]) / len(df)}, Total {len(dfs[split_name])}"
        )

    return dfs
