import math
import multiprocessing

from gm.util import get_obj_from_str

import mindspore.dataset as de


def create_loader(
    data_path,
    rank=0,
    rank_size=1,
    *,
    dataset_config,
    per_batch_size,
    total_step=1000,
    num_parallel_workers=8,
    shuffle=True,
    drop_remainder=True,
    python_multiprocessing=False,
):
    r"""Creates dataloader.

    Applies operations such as transform and batch to the `ms.dataset.Dataset` object
    created by the `create_dataset` function to get the dataloader.

    Args:
        dataset (COCODataset): dataset object created by `create_dataset`.
        per_batch_size (int or function): The number of rows each batch is created with. An
            int or callable object which takes exactly 1 parameter, BatchInfo.
        drop_remainder (bool, optional): Determines whether to drop the last block
            whose data row number is less than batch size (default=False). If True, and if there are less
            than per_batch_size rows available to make the last batch, then those rows will
            be dropped and not propagated to the child node.
        num_parallel_workers (int, optional): Number of workers(threads) to process the dataset in parallel
            (default=None).
        python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker processes. This
            option could be beneficial if the Python operation is computational heavy (default=False).

    Returns:
        BatchDataset, dataset batched.
    """
    dataset = get_obj_from_str(dataset_config["target"])(data_path=data_path, **dataset_config.get("params", dict()))
    batch_collate_fn, dataset_column_names = dataset.collate_fn, dataset.dataset_column_names
    dataset_size = len(dataset)
    num_step_per_epoch = dataset_size // (per_batch_size * rank_size)
    epoch_size = math.ceil(total_step / num_step_per_epoch)

    de.config.set_seed(1236517205 + rank)
    cores = multiprocessing.cpu_count()
    num_parallel_workers = min(int(cores / rank_size), num_parallel_workers)
    print(f"Dataloader num parallel workers: [{num_parallel_workers}]")
    if rank_size > 1:
        ds = de.GeneratorDataset(
            dataset,
            column_names=dataset_column_names,
            num_parallel_workers=min(8, num_parallel_workers),
            shuffle=shuffle,
            python_multiprocessing=python_multiprocessing,
            num_shards=rank_size,
            shard_id=rank,
        )
    else:
        ds = de.GeneratorDataset(
            dataset,
            column_names=dataset_column_names,
            num_parallel_workers=min(32, num_parallel_workers),
            shuffle=shuffle,
            python_multiprocessing=python_multiprocessing,
        )
    ds = ds.batch(
        per_batch_size,
        per_batch_map=batch_collate_fn,
        input_columns=dataset_column_names,
        output_columns=dataset_column_names,
        drop_remainder=drop_remainder,
    )
    ds = ds.repeat(epoch_size)

    return ds
