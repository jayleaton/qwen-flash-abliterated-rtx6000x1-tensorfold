"""recipe soak's verdict on soak-sh/, a synthetic soak.sh log directory, and the hardware profiles' link env."""

import tempfile
import unittest
from pathlib import Path

from recipe_kit import hardware, node
from stand_in import SINGLE
from tensorfold_testkit import soak
from tensorfold_testkit.gates import hashes


class Replay(unittest.TestCase):
    def test_the_soak_log_replays_to_its_verdict(self):
        record = soak.from_soak_sh(SINGLE / "soak-sh")
        rtx = hardware.load("rtx6000x2")
        floors = soak.Floors(rtx.floor_gib, rtx.thermal["hard_c"], rtx.thermal["hard_hold_s"], rtx.power["limit_w"])
        table = hashes.load(SINGLE / "gates/hashes.json")
        verdict = soak.verdict(record, floors, hashes.bench_expected(table, None))
        self.assertEqual(
            verdict["summary"], "STABLE: 4 probes, 8 load rounds; probe hashes constant; failed requests 0"
        )
        want = {name: row["greedy"] for name, row in table["expected"]["*"].items()}
        self.assertEqual(
            verdict["hashes"],
            {
                "cold.2048": [table["cold"]["2048"]["hash"][:12]],
                "decode.chat.x1": [want["gpu-chat-no-think"]],
                "decode.code.x1": [want["fibonacci-raw"]],
            },
        )
        self.assertEqual(verdict["unmeasured"], ["canary", "drain", "memory"])


class Hardware(unittest.TestCase):
    def test_a_roce_link_resolves_to_the_nccl_env_of_its_ranks(self):
        with tempfile.TemporaryDirectory() as sys_root:
            root = Path(sys_root)
            address = ":".join(["0000"] * 5 + ["ffff", "c000", "0207"])
            for netdev, hca, state in (
                ("enp1s0f1np1", "rocep1s0f1", "up"),
                ("enP2p1s0f1np1", "roceP2p1s0f1", "up"),
                ("enp1s0f0np0", "rocep1s0f0", "down"),
            ):
                (root / "class/net" / netdev / "device/infiniband" / hca).mkdir(parents=True)
                (root / "class/net" / netdev / "operstate").write_text(state + "\n")
                port = root / "class/infiniband" / hca / "ports/1"
                for index, gid, kind in ((0, "fe80" + ":0000" * 7, "IB/RoCE v1"), (3, address, "RoCE v2")):
                    (port / "gids").mkdir(parents=True, exist_ok=True)
                    (port / "gid_attrs/types").mkdir(parents=True, exist_ok=True)
                    (port / "gids" / str(index)).write_text(gid + "\n")
                    (port / "gid_attrs/types" / str(index)).write_text(kind + "\n")
            got = node.resolve(hardware.load("gb10x2").env, root)
        want = {
            "NCCL_SOCKET_IFNAME": "enp1s0f1np1",
            "NCCL_IB_HCA": "rocep1s0f1,roceP2p1s0f1",
            "NCCL_IB_GID_INDEX": "3",
            "NCCL_CUMEM_ENABLE": "0",
            "NCCL_CUMEM_HOST_ENABLE": "0",
        }
        self.assertEqual(got, want)

    def test_a_node_without_a_roce_link_refuses(self):
        with tempfile.TemporaryDirectory() as sys_root, self.assertRaises(hardware.ManifestError):
            node.resolve(hardware.load("gb10x2").env, Path(sys_root))


if __name__ == "__main__":
    unittest.main()
