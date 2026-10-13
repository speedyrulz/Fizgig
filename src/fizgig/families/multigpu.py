"""Two (or more) GPUs for one run of the standard trainer (families/train.py), in either of two ways:

    ddp    data parallel: every GPU holds the whole model and trains its own share of each epoch's images. After each
           micro-batch window the trainable adapter's gradients are averaged across the GPUs (an all-reduce of the
           LoRA only - tens to hundreds of MB, never the base), so every GPU takes the same optimizer step and the
           adapters stay identical. An epoch takes about 1/N of the steps: close to N times the speed, at the VRAM of
           one GPU per card.
    split  model parallel: one copy of the model, its blocks divided between two GPUs (the head, the first blocks
           and the output layer on the first, the remaining blocks on the second). Each card holds about half the
           weights, so a precision that needs block swap on one card can train without it. Not faster than one
           card that fits the model - the cards take turns - but much faster than block swap.

DDP needs no launcher: train.py re-runs itself once per GPU (torch.multiprocessing.spawn) and the processes meet over
NCCL (Linux) or Gloo (Windows, which has no NCCL; the gradients then travel through system RAM). Only the first
process writes anything - checkpoints, states, previews, logs; the others train quietly. The GPUs used are the
first N visible ones: CUDA_VISIBLE_DEVICES picks which.

A family offers these in its description (multi_gpu = ("ddp", "split")); "split" also needs the driver's
split_model."""
import datetime
import logging
import os
import socket
import sys

import torch

logger = logging.getLogger(__name__)

MODES = ("off", "ddp", "split")
_ENV_RANK = "FIZGIG_DDP_RANK"


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def gpu_count():
    try:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


# ---- DDP: launching ----------------------------------------------------------------------------------------------
def in_worker():
    """True inside a process launch_ddp started (or one torchrun started)."""
    return _ENV_RANK in os.environ or "LOCAL_RANK" in os.environ


def _worker(rank, world, port, argv, entry):
    os.environ.update({_ENV_RANK: str(rank), "RANK": str(rank), "LOCAL_RANK": str(rank),
                       "WORLD_SIZE": str(world), "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port)})
    sys.argv = list(argv)
    entry()


def launch_ddp(entry, world=2):
    """Run `entry` (the trainer's main) once per GPU and wait for all of them. Returns False when there are fewer
    than `world` GPUs (the caller trains on one instead)."""
    have = gpu_count()
    if have < world:
        logger.warning(f"[multi-gpu] data parallel needs {world} GPUs, {have} visible - training on one GPU")
        return False
    import torch.multiprocessing as mp
    port = _free_port()
    logger.info(f"[multi-gpu] data parallel: starting {world} training processes (one per GPU)")
    mp.spawn(_worker, args=(world, port, list(sys.argv), entry), nprocs=world, join=True)
    return True


# ---- DDP: inside a worker ------------------------------------------------------------------------------------------
class DDP:
    """The collective side of a data-parallel run, one per process. Every method is a no-op shape-wise on a single
    process, but train.py only builds one when the run is data parallel."""

    def __init__(self):
        import torch.distributed as dist
        self.dist = dist
        self.rank = int(os.environ.get("RANK", os.environ.get(_ENV_RANK, "0")))
        self.local_rank = int(os.environ.get("LOCAL_RANK", self.rank))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        # NCCL where it exists; Windows builds of PyTorch have only Gloo, which reduces CPU tensors
        self.backend = "nccl" if sys.platform != "win32" and dist.is_nccl_available() else "gloo"
        if sys.platform == "win32":
            os.environ.setdefault("USE_LIBUV", "0")      # Windows builds' TCPStore has no libuv
        # previews, checkpoint saves and caption fixes run on the first process while the others wait at a barrier:
        # the timeout has to outlast the longest of them, not NCCL's 10-minute default
        kw = {"device_id": self.device} if self.backend == "nccl" else {}
        try:
            dist.init_process_group(self.backend, rank=self.rank, world_size=self.world,
                                    timeout=datetime.timedelta(hours=6), **kw)
        except TypeError:                                  # older PyTorch: no device_id
            dist.init_process_group(self.backend, rank=self.rank, world_size=self.world,
                                    timeout=datetime.timedelta(hours=6))
        self.main = self.rank == 0
        if not self.main:
            # one console: the first process speaks for the run; the others only warn
            logging.getLogger().setLevel(logging.WARNING)
        logger.info(f"[multi-gpu] data parallel over {self.world} GPUs ({self.backend}): "
                    + ", ".join(torch.cuda.get_device_name(i) for i in range(self.world)))

    # --- collectives
    def _reduce(self, t, op):
        if self.backend == "gloo" and t.is_cuda:
            c = t.detach().cpu()
            self.dist.all_reduce(c, op=op)
            t.copy_(c)
        else:
            self.dist.all_reduce(t, op=op)
        return t

    def mean(self, x):
        """The mean of a Python number across the processes."""
        t = torch.tensor([float(x)], dtype=torch.float64, device=self.device)
        return float(self._reduce(t, self.dist.ReduceOp.SUM).item()) / self.world

    @torch.no_grad()
    def average_(self, t):
        """Average one tensor across the processes, in place (a fused fine-tune step's gradient)."""
        self._reduce(t, self.dist.ReduceOp.SUM)
        t.div_(self.world)
        return t

    def minimum(self, x):
        t = torch.tensor([float(x)], dtype=torch.float64, device=self.device)
        return float(self._reduce(t, self.dist.ReduceOp.MIN).item())

    def decide(self, flag):
        """The first process's answer, on every process (a file it checked, which the others might see a moment
        later or earlier)."""
        t = torch.tensor([1.0 if (flag if self.main else False) else 0.0], device=self.device)
        self._broadcast(t)
        return bool(t.item() > 0)

    def _broadcast(self, t):
        if self.backend == "gloo" and t.is_cuda:
            c = t.detach().cpu()
            self.dist.broadcast(c, src=0)
            t.copy_(c)
        else:
            self.dist.broadcast(t, src=0)

    def barrier(self):
        if self.backend == "nccl":
            self.dist.barrier(device_ids=[self.local_rank])
        else:
            self.dist.barrier()

    @torch.no_grad()
    def average_gradients(self, params, bucket_elems=1 << 25):
        """Average every trainable parameter's gradient across the processes, in flat buckets (one all-reduce per
        ~128 MB of fp32). A parameter without a gradient (a skipped step, a block frozen this step) counts as
        zero, so every process reduces the same list."""
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
        params = list(params)              # every process the same list, whatever this step froze
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        buckets, cur, n = [], [], 0
        for p in params:
            cur.append(p)
            n += p.numel()
            if n >= bucket_elems:
                buckets.append(cur)
                cur, n = [], 0
        if cur:
            buckets.append(cur)
        for b in buckets:
            by_dtype = {}
            for p in b:
                by_dtype.setdefault(p.grad.dtype, []).append(p)
            for ps in by_dtype.values():
                grads = [p.grad for p in ps]
                flat = _flatten_dense_tensors(grads)
                self._reduce(flat, self.dist.ReduceOp.SUM)
                flat.div_(self.world)
                for g, r in zip(grads, _unflatten_dense_tensors(flat, grads)):
                    g.copy_(r)

    @torch.no_grad()
    def broadcast_parameters(self, params):
        """Every process takes the first process's values (the adapter's start, a resumed state, and once an epoch
        as a guard against any drift between the copies)."""
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
        params = list(params)
        by = {}
        for p in params:
            by.setdefault((p.device, p.dtype), []).append(p)
        for ps in by.values():
            flat = _flatten_dense_tensors([p.data for p in ps])
            self._broadcast(flat)
            for p, r in zip(ps, _unflatten_dense_tensors(flat, [p.data for p in ps])):
                p.data.copy_(r)

    def close(self):
        try:
            if self.dist.is_initialized():
                self.dist.destroy_process_group()
        except Exception:
            pass


# ---- split: one model across two GPUs ------------------------------------------------------------------------------
def split_devices(n=2):
    return [torch.device("cuda", i) for i in range(n)]


def free_gb(device):
    free, _ = torch.cuda.mem_get_info(device)
    return free / 1024 ** 3


def auto_split_at(n_blocks, block_gb, head_gb, free0, free1, reserve0=2.5, reserve1=1.0):
    """Where the second GPU's blocks start: the blocks shared in proportion to the room each card has for them, after
    the first card's own load (the head, the text-fusion stack, the output layer and the preview VAE) and a working
    reserve on each. At least one block on each card."""
    room0 = max(0.0, free0 - head_gb - reserve0)
    room1 = max(0.0, free1 - reserve1)
    if room0 + room1 <= 0 or block_gb <= 0:
        return n_blocks // 2
    at = round(n_blocks * room0 / (room0 + room1))
    return max(1, min(n_blocks - 1, at))


def is_split(module):
    return getattr(module, "_fizgig_split", None) is not None


def module_gb(m):
    """A module's resident size: its parameters and buffers, plus NF4's packed weights (plain attributes)."""
    n = sum(t.numel() * t.element_size() for t in list(m.parameters()) + list(m.buffers()))
    for sub in m.modules():
        packed = getattr(sub, "_nf4_packed", None)
        if torch.is_tensor(packed):
            n += packed.numel() * packed.element_size()
    return n / 1024 ** 3


def _to(m, device):
    m.to(device)
    if any(getattr(sub, "_is_nf4", False) for sub in m.modules()):
        from fizgig.modules.nf4 import move_nf4_to_device
        move_nf4_to_device(m, device)


def split_model(driver, dit, devices, at=None):
    """Divide `dit` (on the CPU, or anywhere) between two GPUs: its transformer blocks from `at` on to the second,
    everything else to the first. at=None picks the split from the free VRAM of each card. The model keeps the plan
    (dit._fizgig_split) - its forward moves the block inputs across at `at`, and quant.move puts the parts back
    after a park. Returns `at`."""
    blocks = driver.split_blocks(dit)
    if blocks is None:
        raise RuntimeError(f"{driver.description.display_name} cannot split its model across GPUs")
    d0, d1 = (torch.device(d) for d in devices[:2])
    n = len(blocks)
    block_gb = max(module_gb(b) for b in blocks)
    head_gb = module_gb(dit) - sum(module_gb(b) for b in blocks)
    if at is None:
        at = auto_split_at(n, block_gb, head_gb, free_gb(d0), free_gb(d1))
        how = "auto, from each card's free VRAM"
    else:
        at = max(1, min(n - 1, int(at)))
        how = "as set"
    owner = {id(b): (d0 if i < at else d1) for i, b in enumerate(blocks)}

    def _place(model):
        # the blocks first, each to its card; then the rest of the model to the first card without touching them
        for b in blocks:
            _to(b, owner[id(b)])
        parent, name = _blocks_owner(model, blocks)
        setattr(parent, name, torch.nn.ModuleList())
        try:
            _to(model, d0)
        finally:
            setattr(parent, name, blocks)
        torch.cuda.empty_cache()

    dit._fizgig_split = {"at": at, "devices": (d0, d1), "place": _place}
    _place(dit)
    logger.info(f"[multi-gpu] model split across 2 GPUs ({how}): blocks 0-{at - 1} and the input/output layers on "
                f"{torch.cuda.get_device_name(d0)} (cuda:{d0.index}), blocks {at}-{n - 1} on "
                f"{torch.cuda.get_device_name(d1)} (cuda:{d1.index}); ~{block_gb:.2f} GB per block, "
                f"{head_gb:.2f} GB outside the blocks")
    return at


def _blocks_owner(model, blocks):
    for name, mod in model.named_modules():
        for cname, child in mod.named_children():
            if child is blocks:
                return mod, cname
    raise RuntimeError("the split model's block list is no longer in the model")


def place(dit):
    """Put a split model's parts back on their GPUs (after a park on the CPU)."""
    dit._fizgig_split["place"](dit)


def split_peak_note(devices):
    """'cuda:0 x GB, cuda:1 y GB' - the peak reserved VRAM of each card, for the epoch line."""
    return ", ".join(f"cuda:{torch.device(d).index} {torch.cuda.max_memory_reserved(d) / 1024 ** 3:.1f} GB"
                     for d in devices)
