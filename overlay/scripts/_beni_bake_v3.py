# Bake v3: transform the RAW upstream BENI OBJs to body frame by hand.
#
# MotrixSim ignores MJCF joint stiffness/springref and mesh refpos/refquat is
# fragile across engines, so this script rewrites the eight VIS meshes with
# MuJoCo's ref transform applied to the vertices only (v' = R(refquat)^-1 *
# (v - refpos)), preserving every vt/f line byte-exact.
#
# Run from anywhere; paths are derived from this file's location:
#   <repo>/motrix_envs/src/motrix_envs/robot/assets/beni
# where <repo> is the MotrixLab checkout containing this script under scripts/.

import pathlib

import numpy as np

# Locate the beni asset dir relative to this script (scripts/ -> repo root).
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
ASSET = REPO / "motrix_envs" / "src" / "motrix_envs" / "robot" / "assets" / "beni"

RAW = ASSET / "meshes"
BAKED = ASSET / "meshes_baked"
BAKED.mkdir(exist_ok=True)

REFS = {
    "BASE_VIS": ([0.0, 0.11, 0.0], [0.707106781187, -0.707106781187, 0.0, 0.0]),
    "HEAD_VIS": ([0.0, 0.1469, 0.0], [0.707106781187, -0.707106781187, 0.0, 0.0]),
    "L1_VIS": ([0.0, 0.11, -0.048], [0.69916673425, -0.69916673425, -0.10566871684, 0.10566871684]),
    "L2_VIS": ([-0.0668735542388, 0.0893135855337, -0.072], [0.651288474746, -0.651288474746, -0.275360350565, 0.275360350565]),
    "LW_VIS": ([-0.0130718474213, 0.0370605823327, -0.08], [0.651288474746, -0.651288474746, -0.275360350565, 0.275360350565]),
    "R1_VIS": ([0.0, 0.11, 0.048], [0.69916673425, -0.69916673425, -0.10566871684, 0.10566871684]),
    "R2_VIS": ([-0.0668735542388, 0.0893135855337, 0.072], [0.651288474746, -0.651288474746, -0.275360350565, 0.275360350565]),
    "RW_VIS": ([-0.0130718474213, 0.0370605823327, 0.08], [0.651288474746, -0.651288474746, -0.275360350565, 0.275360350565]),
}
FILE_OF = {
    "BASE_VIS": "VIS_base_link_00_BASE_VIS.obj",
    "HEAD_VIS": "VIS_head_00_HEAD_VIS.obj",
    "L1_VIS": "VIS_L1_Link_00_L1_VIS.obj",
    "L2_VIS": "VIS_L2_Link_00_L1_VIS.obj",
    "LW_VIS": "VIS_LW_Link_00_LW_VIS.obj",
    "R1_VIS": "VIS_R1_Link_00_R1_VIS.obj",
    "R2_VIS": "VIS_R2_Link_00_R2_VIS.obj",
    "RW_VIS": "VIS_RW_Link_00_RW_VIS.obj",
}


def quat_conj_mul_vec(q, v):
    """Apply inverse rotation of quat q (w,x,y,z) to vector v."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    # inverse rotation = rotation by conjugate
    qc = np.array([w, -x, -y, -z])
    w2, x2, y2, z2 = qc
    uv = np.cross([x2, y2, z2], v)
    uuv = np.cross([x2, y2, z2], uv)
    return v + 2 * (w2 * uv + uuv)


if not (RAW / FILE_OF["BASE_VIS"]).exists():
    raise SystemExit(
        f"raw meshes not found at {RAW}\n"
        "This script needs the ORIGINAL upstream OBJs (meshes/); the shipped "
        "meshes_baked/ output is already generated, so you only need this to "
        "re-bake from scratch."
    )

for name, (refpos, refquat) in REFS.items():
    src = RAW / FILE_OF[name]
    out = BAKED / FILE_OF[name]
    rp = np.asarray(refpos)
    lines_out = []
    n_v = n_vt = n_f = 0
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            p = line.split()
            v = np.array([float(p[1]), float(p[2]), float(p[3])])
            vb = quat_conj_mul_vec(refquat, v - rp)
            lines_out.append(f"v {vb[0]:.6f} {vb[1]:.6f} {vb[2]:.6f}")
            n_v += 1
        elif line.startswith(("vt ", "f ", "vn ")):
            lines_out.append(line)
            if line.startswith("vt "):
                n_vt += 1
            elif line.startswith("f "):
                n_f += 1
    out.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print(f"{name}: v={n_v} vt={n_vt} f={n_f} -> {out.name}")
print("done (beni.xml already points meshdir to meshes_baked, no refs)")
