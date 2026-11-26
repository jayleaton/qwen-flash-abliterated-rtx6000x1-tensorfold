"""The values a hardware profile names "@name", read on the node from sysfs at launch and never written to a repo."""

from __future__ import annotations

from pathlib import Path

from .manifest import ManifestError

NAMES = ("rdma-netdev", "rdma-devices", "roce-v2-gid")
V2_IPV4 = "0000:0000:0000:0000:0000:ffff:"


def links(sys_root: Path) -> list[tuple[str, str, int]]:
    """(netdev, RDMA device, GID index) for each up netdev with an RDMA device whose port holds a RoCE v2 IPv4 GID
    (an all-zero or v1 entry makes NCCL fail about a minute in), by name."""
    found = []
    nets = sys_root / "class" / "net"
    for net in sorted(nets.iterdir() if nets.is_dir() else (), key=lambda p: p.name.lower()):
        ib = net / "device" / "infiniband"
        if not ib.is_dir() or _read(net / "operstate") != "up" or not (hcas := sorted(p.name for p in ib.iterdir())):
            continue
        port = sys_root / "class" / "infiniband" / hcas[0] / "ports" / "1"
        gids = sorted((port / "gids").iterdir(), key=lambda p: int(p.name)) if (port / "gids").is_dir() else []
        for gid in gids:
            if _read(gid).startswith(V2_IPV4) and "v2" in _read(port / "gid_attrs" / "types" / gid.name).lower():
                found.append((net.name, hcas[0], int(gid.name)))
                break
    return found


def resolve(env: dict[str, str], sys_root: Path = Path("/sys")) -> dict[str, str]:
    """env with each "@name" replaced: the first such link's netdev and GID index, and every RDMA device with a RoCE
    v2 GID at that index (a Spark's twin PCIe link of the same port is a second rail)."""
    wanted = sorted(name for name, value in env.items() if value.startswith("@"))
    if not wanted:
        return dict(env)
    found = links(sys_root)
    if not found:
        raise ManifestError(
            f"{', '.join(wanted)}: this node has no up netdev with an RDMA device and a RoCE v2 IPv4 GID; "
            "set them in the environment"
        )
    netdev, _, gid = found[0]
    values = {
        "rdma-netdev": netdev,
        "rdma-devices": ",".join(hca for _, hca, index in found if index == gid),
        "roce-v2-gid": str(gid),
    }
    return {name: values[value[1:]] if value.startswith("@") else value for name, value in env.items()}


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""
