#!/usr/bin/env python3
"""
Generate figures/datagen_overview.png

A visual explanation of the synthetic proof-chain data generation pipeline:
  Backward search tree  →  Linearisation  →  Chain compression  →  NL rendering

Run:
    python3 scripts/gen_datagen_figure.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Output path ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
OUT = os.path.join(ROOT, "figures", "datagen_overview.png")

# ── Palette ─────────────────────────────────────────────────────────────────────
BG       = "#F0F2F5"
PBG      = "#FFFFFF"

GOAL_F   = "#C0392B"   # red    – goal / conclusion nodes
GOAL_E   = "#922B21"
RULE_F   = "#1A5276"   # navy   – ND rule boxes
RULE_E   = "#154360"
INIT_F   = "#1A6B3C"   # green  – initial premises (leaves)
INIT_E   = "#145A32"
SUB_F    = "#6C3483"   # purple – sub-goals (intermediate targets)
SUB_E    = "#512E5F"
SEG_F    = "#935116"   # amber  – compressed segment boundary
SEG_E    = "#784212"
TAG_F    = "#0E6655"   # teal   – rendered output tags
ABS_F    = "#797D7F"   # grey   – absorbed intermediate

DARK     = "#1C2833"
MID      = "#5D6D7E"
LIGHT    = "#BDC3C7"
WHITE    = "#FFFFFF"

# ── Canvas ───────────────────────────────────────────────────────────────────────
FW, FH = 22, 15
fig = plt.figure(figsize=(FW, FH), dpi=150)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis("off")
ax.set_facecolor(BG)


# ── Drawing helpers ──────────────────────────────────────────────────────────────
def rbox(cx, cy, w, h, text, fc, ec=WHITE, tc=WHITE,
         fs=9.5, bold=False, alpha=1.0, ls="-", lw=1.8,
         z=3, mono=False, pad=0.18, va_text="center"):
    patch = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={pad}",
        fc=fc, ec=ec, lw=lw, ls=ls, alpha=alpha, zorder=z,
    )
    ax.add_patch(patch)
    fw = "bold" if bold else "normal"
    ff = "monospace" if mono else "DejaVu Sans"
    ax.text(cx, cy, text, ha="center", va=va_text, fontsize=fs,
            color=tc, fontweight=fw, fontfamily=ff,
            multialignment="center", zorder=z + 1, alpha=alpha,
            linespacing=1.4)
    return patch


def arr(x1, y1, x2, y2, color=MID, lw=2.0, hw=0.27, hl=0.36,
        label=None, lside="right", conn="arc3,rad=0", z=6, dash=False):
    ls = (0, (5, 3)) if dash else "solid"
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1), zorder=z,
        arrowprops=dict(
            arrowstyle=f"->, head_width={hw}, head_length={hl}",
            color=color, lw=lw, ls=ls, connectionstyle=conn,
        ),
    )
    if label:
        lx = x1 + (x2 - x1) * 0.48
        ly = y1 + (y2 - y1) * 0.48
        dx = 0.18 if lside == "right" else -0.18
        ha = "left" if lside == "right" else "right"
        ax.text(lx + dx, ly, label, ha=ha, va="center", fontsize=7.5,
                color=color, style="italic", zorder=z + 1)


def hline(x1, x2, y, color=LIGHT, lw=1.0, ls="--"):
    ax.plot([x1, x2], [y, y], color=color, lw=lw, ls=ls, zorder=1)


def section_hdr(cx, y, text, fs=11):
    ax.text(cx, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold", color=DARK, zorder=8)


# ═══════════════════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════════════════
ax.text(FW / 2, FH - 0.35,
        "Synthetic Proof-Chain Data Generation",
        ha="center", va="top", fontsize=17, fontweight="bold",
        color=DARK, zorder=10)
ax.text(FW / 2, FH - 0.98,
        "Backward natural deduction search  ·  Topological linearisation  "
        "·  Chain compression  ·  NL rendering",
        ha="center", va="top", fontsize=9.5, color=MID, zorder=10)

# Divider between Phase 1 and bottom panels
DIV_Y = 5.75
hline(0.4, FW - 0.4, DIV_Y, color=LIGHT, lw=1.8, ls="-")

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — BACKWARD SEARCH TREE  (y: 5.9 → 13.3)
# ═══════════════════════════════════════════════════════════════════════════════
section_hdr(FW / 2, DIV_Y + 0.42,
            "① Backward Search Tree  (chain_generator._justify)", fs=11)

# ── Node positions ──────────────────────────────────────────────────────────────
#  Example: derive R ("plants grow") from initial premises P→Q, Q→R, P
#  Goal R via IMPLIES_ELIM needs: Q→R (initial) and Q (sub-goal)
#  Sub-goal Q via IMPLIES_ELIM needs: P→Q (initial) and P (initial)

G_GOAL = (11.0, 13.00)   # Goal: R  "plants grow"
G_R1   = (11.0, 11.75)  # Rule 1: IMPLIES_ELIM → derives R
G_Q    = ( 7.2, 10.30)  # Sub-goal: Q  "soil is wet"
G_QR   = (15.0, 10.30)  # Initial: Q→R  "if soil wet, plants grow"
G_R2   = ( 7.2,  8.95)  # Rule 2: IMPLIES_ELIM → derives Q
G_PQ   = ( 4.5,  7.60)  # Initial: P→Q  "if it rains, soil is wet"
G_P    = ( 9.8,  7.60)  # Initial: P   "it rains"

NW_GOAL  = 4.5;  NH = 0.88
NW_RULE  = 4.0
NW_INIT  = 4.2
NW_SUB   = 3.6

# Draw nodes
rbox(*G_GOAL, NW_GOAL, NH,
     '"plants grow"\n(Goal:  R)', GOAL_F, GOAL_E, WHITE, fs=9.5, bold=True)

rbox(*G_R1, NW_RULE, NH,
     "IMPLIES_ELIM\nneeds  [Q→R]  and  [Q]", RULE_F, RULE_E, WHITE, fs=8.5)

rbox(*G_Q, NW_SUB, NH,
     '"soil is wet"\n(Sub-goal:  Q)', SUB_F, SUB_E, WHITE, fs=9.0)

rbox(*G_QR, NW_INIT, NH,
     '"if soil wet, plants grow"\n(Initial Premise:  Q→R)', INIT_F, INIT_E, WHITE, fs=8.5)

rbox(*G_R2, NW_RULE, NH,
     "IMPLIES_ELIM\nneeds  [P→Q]  and  [P]", RULE_F, RULE_E, WHITE, fs=8.5)

rbox(*G_PQ, NW_INIT, NH,
     '"if it rains, soil is wet"\n(Initial Premise:  P→Q)', INIT_F, INIT_E, WHITE, fs=8.5)

rbox(*G_P, 3.0, NH,
     '"it rains"\n(Initial Premise:  P)', INIT_F, INIT_E, WHITE, fs=8.5)

# ── Tree arrows ─────────────────────────────────────────────────────────────────
# Goal → Rule1
arr(G_GOAL[0], G_GOAL[1] - NH/2,
    G_R1[0],   G_R1[1]   + NH/2, color=GOAL_F, lw=2.2)

# Rule1 → Q sub-goal  (left branch)
arr(G_R1[0] - 1.0, G_R1[1] - NH/2,
    G_Q[0],         G_Q[1]   + NH/2, color=RULE_F, lw=2.2)

# Rule1 → Q→R initial  (right branch)
arr(G_R1[0] + 1.5, G_R1[1] - NH/2,
    G_QR[0],        G_QR[1]  + NH/2, color=RULE_F, lw=2.2)

# Q sub-goal → Rule2
arr(G_Q[0], G_Q[1] - NH/2,
    G_R2[0], G_R2[1] + NH/2, color=SUB_F, lw=2.2)

# Rule2 → P→Q initial  (left branch)
arr(G_R2[0] - 1.0, G_R2[1] - NH/2,
    G_PQ[0],        G_PQ[1]  + NH/2, color=RULE_F, lw=2.2)

# Rule2 → P initial  (right branch)
arr(G_R2[0] + 1.0, G_R2[1] - NH/2,
    G_P[0],         G_P[1]   + NH/2, color=RULE_F, lw=2.2)

# ── Annotation: query text beside arrows ────────────────────────────────────────
ax.text(G_GOAL[0] + 0.35, (G_GOAL[1] + G_R1[1]) / 2,
        "_justify(R, depth=2):\n'which rule derives R?'",
        ha="left", va="center", fontsize=8.0, color=GOAL_F,
        style="italic", zorder=9)

ax.text(G_Q[0] + 0.35, (G_Q[1] + G_R2[1]) / 2,
        "_justify(Q, depth=1):\n'which rule derives Q?'",
        ha="left", va="center", fontsize=8.0, color=SUB_F,
        style="italic", zorder=9)

# ── Leaf annotations ────────────────────────────────────────────────────────────
for gnode in [G_QR, G_PQ, G_P]:
    ax.text(gnode[0], gnode[1] - NH/2 - 0.26,
            "▲ leaf → becomes initial premise",
            ha="center", va="top", fontsize=7.0, color=INIT_F,
            style="italic", zorder=9)

# ── Backward / Forward direction indicators (sides) ─────────────────────────────
# Left: backward search direction (top-down arrow)
ax.annotate("", xy=(1.4, 7.5), xytext=(1.4, 12.8), zorder=9,
            arrowprops=dict(arrowstyle="->, head_width=0.38, head_length=0.45",
                            color=GOAL_F, lw=2.8))
ax.text(0.72, 10.2,
        "Backward\nsearch\ndirection\n\n_justify()\nrecurses\ndown",
        ha="center", va="center", fontsize=8.5, color=GOAL_F,
        fontweight="bold", rotation=90, zorder=9)

# Right: forward proof direction (bottom-up arrow)
ax.annotate("", xy=(FW - 1.4, 12.8), xytext=(FW - 1.4, 7.5), zorder=9,
            arrowprops=dict(arrowstyle="->, head_width=0.38, head_length=0.45",
                            color=INIT_F, lw=2.8))
ax.text(FW - 0.72, 10.2,
        "Forward\nproof\nexecution\n\npremises\n→ derived\n→ goal",
        ha="center", va="center", fontsize=8.5, color=INIT_F,
        fontweight="bold", rotation=90, zorder=9)

# ── Legend ───────────────────────────────────────────────────────────────────────
LX, LY0 = 19.5, 8.80
legend_bg = mpatches.FancyBboxPatch(
    (LX - 1.72, LY0 - 3.45), 3.44, 4.10,
    boxstyle="round,pad=0.18", fc=PBG, ec=LIGHT, lw=1.4, zorder=6)
ax.add_patch(legend_bg)
for i, (lbl, fc, ec) in enumerate([
    ("Goal (conclusion)", GOAL_F, GOAL_E),
    ("ND Rule applied",   RULE_F, RULE_E),
    ("Sub-goal (recurse)",SUB_F,  SUB_E),
    ("Initial Premise",   INIT_F, INIT_E),
]):
    rbox(LX, LY0 - i * 0.80, 3.0, 0.58, lbl, fc, ec, WHITE, fs=8.0, z=7)
ax.text(LX, LY0 + 0.52, "Legend", ha="center", va="bottom",
        fontsize=9.5, fontweight="bold", color=DARK, zorder=8)

# ── Z3 backstop badge ────────────────────────────────────────────────────────────
Z3X, Z3Y = 19.5, 13.1
rbox(Z3X, Z3Y, 3.2, 0.82,
     "Z3 SMT Backstop\n✓ verifies full chain", "#2C3E50", "#1A252F", WHITE, fs=8.5, z=8)
# dashed arc from leaf premises area to badge
ax.annotate("", xy=(Z3X - 0.5, Z3Y - 0.42), xytext=(9.8, 7.2), zorder=7,
            arrowprops=dict(
                arrowstyle="->, head_width=0.22, head_length=0.28",
                color="#2C3E50", lw=1.6, ls=(0, (4, 3)),
                connectionstyle="arc3,rad=-0.28",
            ))
ax.text(17.3, 10.6, "if rejected:\nretry", ha="center", va="center",
        fontsize=7.5, color="#2C3E50", style="italic", zorder=8)


# ═══════════════════════════════════════════════════════════════════════════════
# BOTTOM PANELS  (y: 0.25 → 5.55)
# ═══════════════════════════════════════════════════════════════════════════════
PY1, PY2 = 0.28, 5.52
PCY = (PY1 + PY2) / 2  # 2.9

# Panel x ranges
P2X1, P2X2 = 0.30,  7.15   # Phase 2: linearised steps
P3X1, P3X2 = 7.45, 14.35   # Phase 3: compression
P4X1, P4X2 = 14.65, 21.7   # Phase 4: output

for (px1, px2) in [(P2X1, P2X2), (P3X1, P3X2), (P4X1, P4X2)]:
    bg = mpatches.FancyBboxPatch(
        (px1, PY1), px2 - px1, PY2 - PY1,
        boxstyle="round,pad=0.15", fc=PBG, ec=LIGHT, lw=1.8, zorder=0, alpha=0.92)
    ax.add_patch(bg)

# Inter-panel flow arrows
for (x1, x2) in [(P2X2 + 0.05, P3X1 - 0.05), (P3X2 + 0.05, P4X1 - 0.05)]:
    arr(x1, PCY, x2, PCY, color=MID, lw=2.8, hw=0.32, hl=0.42)

# Large downward arrow from Phase-1 tree into Phase-2 panel
arr(FW / 2 - 6.5, DIV_Y - 0.04, (P2X1 + P2X2) / 2, PY2 + 0.04,
    color=MID, lw=2.2, hw=0.28, hl=0.38, conn="arc3,rad=0.18")


# ── PHASE 2: Topological Linearisation ─────────────────────────────────────────
P2CX = (P2X1 + P2X2) / 2
section_hdr(P2CX, PY2 - 0.32, "② Topological Linearisation", fs=10)

BW = 6.1
BH = 1.10

# Step 1 box
S1Y = 4.22
rbox(P2CX, S1Y, BW, BH,
     "Step 1  (layer 0)\nIMPLIES_ELIM\n[P→Q ,  P]  ──▶  Q", RULE_F, RULE_E, WHITE, fs=9.0)

# Middle annotation: "Q = intermediate"
IM_Y = 3.2
ax.text(P2CX, IM_Y,
        "Q  =  intermediate formula\n(produced by Step 1, consumed by Step 2)",
        ha="center", va="center", fontsize=8.0, color=SUB_F, fontweight="bold",
        zorder=5)

arr(P2CX, S1Y - BH / 2 - 0.03, P2CX, IM_Y + 0.35,
    color=SUB_F, lw=1.9, hw=0.20, hl=0.26)
arr(P2CX, IM_Y - 0.35, P2CX, 2.22 + BH / 2 + 0.03,
    color=SUB_F, lw=1.9, hw=0.20, hl=0.26)

# Step 2 box
S2Y = 2.05
rbox(P2CX, S2Y, BW, BH,
     "Step 2  (layer 1)\nIMPLIES_ELIM\n[Q→R ,  Q]  ──▶  R", RULE_F, RULE_E, WHITE, fs=9.0)

ax.text(P2CX, PY1 + 0.38,
        "steps.sort(key=lambda s: (s.layer, s.step_id))",
        ha="center", va="center", fontsize=7.5, color=MID,
        style="italic", fontfamily="monospace", zorder=5)


# ── PHASE 3: Chain Compression ──────────────────────────────────────────────────
P3CX = (P3X1 + P3X2) / 2
section_hdr(P3CX, PY2 - 0.32, "③ Chain Compression  (max_n = 2)", fs=10)

# Outer dashed group box = CompressedSegment
GW, GH = 5.8, 3.8
GCY = 2.80
group = mpatches.FancyBboxPatch(
    (P3CX - GW / 2, GCY - GH / 2), GW, GH,
    boxstyle="round,pad=0.22",
    fc="#FFF6E8", ec=SEG_F, lw=2.8, ls=(0, (6, 3)), zorder=2)
ax.add_patch(group)
ax.text(P3CX, GCY + GH / 2 + 0.28, "CompressedSegment",
        ha="center", va="bottom", fontsize=9.5, color=SEG_F,
        fontweight="bold", zorder=5)

# Step 1 inside (faded – its intermediate output Q vanishes)
rbox(P3CX, GCY + 1.12, 5.0, 0.88,
     "Step 1:  [P→Q, P]  ──▶  Q", RULE_F, RULE_E, WHITE, fs=8.5, alpha=0.32, z=3)

# Absorbed intermediate marker
ABS_Y = GCY - 0.05
rbox(P3CX, ABS_Y, 4.2, 0.72,
     "Q  (intermediate — absorbed)", ABS_F, ABS_F, WHITE, fs=8.5, alpha=0.48,
     ls=(0, (3, 2)), z=3)
# Red ✕ on top
ax.text(P3CX + 2.35, ABS_Y, "✕",
        ha="center", va="center", fontsize=18, color=GOAL_F,
        fontweight="bold", zorder=8)
# Strikethrough line
ax.plot([P3CX - 1.85, P3CX + 1.85], [ABS_Y, ABS_Y],
        color=GOAL_F, lw=2.2, zorder=8, alpha=0.7)

# Step 2 inside (faded)
rbox(P3CX, GCY - 1.18, 5.0, 0.88,
     "Step 2:  [Q→R, Q]  ──▶  R", RULE_F, RULE_E, WHITE, fs=8.5, alpha=0.32, z=3)

# ── Leaf inputs entering from the left ───────────────────────────────────────────
LFX = P3X1 + 0.62   # x of leaf label column
leaf_labels = [
    (GCY + 1.12, "P→Q", INIT_F),
    (GCY + 0.35, "P",   INIT_F),
    (GCY - 1.18, "Q→R", INIT_F),
]
ax.text(LFX, GCY + 1.75, "Leaf\ninputs:", ha="center", va="center",
        fontsize=8.0, color=INIT_F, style="italic", fontweight="bold")
for (ly, lbl, col) in leaf_labels:
    ax.text(LFX, ly, lbl, ha="center", va="center",
            fontsize=9.0, color=col, fontweight="bold", zorder=6)
    arr(LFX + 0.42, ly, P3CX - GW / 2 - 0.04, ly,
        color=col, lw=1.6, hw=0.18, hl=0.24)

# ── Output from the right ────────────────────────────────────────────────────────
OUTX = P3X2 - 0.62
ax.text(OUTX, GCY + 0.45, "R", ha="center", va="center",
        fontsize=11, color=GOAL_F, fontweight="bold", zorder=6)
ax.text(OUTX, GCY - 0.12, "Output\n(conclusion)", ha="center", va="center",
        fontsize=7.5, color=GOAL_F, style="italic", zorder=6)
arr(P3CX + GW / 2 + 0.04, GCY, OUTX - 0.30, GCY,
    color=GOAL_F, lw=2.0, hw=0.24, hl=0.32)

ax.text(P3CX, PY1 + 0.38,
        "Leaf inputs: {P→Q, P, Q→R}   ·   Intermediate Q absorbed (not shown in trace)",
        ha="center", va="center", fontsize=7.5, color=MID, style="italic", zorder=5)


# ── PHASE 4: NL Rendering → Training JSONL ──────────────────────────────────────
P4CX = (P4X1 + P4X2) / 2
section_hdr(P4CX, PY2 - 0.32, "④ NL Rendering  →  Training JSONL", fs=10)

# Sub-divider at y=3.1
hline(P4X1 + 0.35, P4X2 - 0.35, 3.10, color=LIGHT, lw=1.0, ls="--")

# ── Stage 0 (top sub-panel) ──────────────────────────────────────────────────────
ax.text(P4CX, PY2 - 0.68,
        "Stage 0  —  conclusion only (no proof trace)",
        ha="center", va="top", fontsize=8.5, color=MID, fontweight="bold")

s0 = (
    "<PREMISE> {if it rains, soil is wet}     </PREMISE>\n"
    "<PREMISE> it rains                        </PREMISE>\n"
    "<PREMISE> {if soil is wet, plants grow}   </PREMISE>\n"
    "<CONCLUSION> plants grow                  </CONCLUSION>"
)
ax.text(P4CX, 4.12, s0,
        ha="center", va="center", fontsize=8.2, color=TAG_F,
        fontfamily="monospace", multialignment="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="#E8F8F5", ec=TAG_F, lw=1.3),
        zorder=4, linespacing=1.55)

# ── Stage 1 (bottom sub-panel) ──────────────────────────────────────────────────
ax.text(P4CX, 3.04,
        "Stage 1  —  proof trace (compression applied)",
        ha="center", va="top", fontsize=8.5, color=MID, fontweight="bold")

# With compression, the intermediate conclusion "soil is wet" is ABSENT.
# Show side-by-side: without compression (left) vs with compression (right)
s1_nocomp = (
    "# max_n = 1 (no compression)\n"
    "<PREMISE> {if it rains, …} </PREMISE>\n"
    "<PREMISE> it rains         </PREMISE>\n"
    "<CONCLUSION> soil is wet   </CONCLUSION>\n"   # ← intermediate shown
    "<PREMISE> soil is wet      </PREMISE>\n"
    "<PREMISE> {if soil wet, …} </PREMISE>\n"
    "<CONCLUSION> plants grow   </CONCLUSION>"
)
s1_comp = (
    "# max_n = 2 (compression)\n"
    "<PREMISE> {if it rains, …} </PREMISE>\n"
    "<PREMISE> it rains         </PREMISE>\n"
    "<PREMISE> {if soil wet, …} </PREMISE>\n"
    "# 'soil is wet' implicit ✕\n"
    "<CONCLUSION> plants grow   </CONCLUSION>"
)

# Left: no compression
ax.text(P4CX - 1.85, 1.82, s1_nocomp,
        ha="center", va="center", fontsize=7.5, color=SUB_F,
        fontfamily="monospace", multialignment="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="#F5EEF8", ec=SUB_F, lw=1.2),
        zorder=4, linespacing=1.45)

# Right: with compression
ax.text(P4CX + 1.95, 1.82, s1_comp,
        ha="center", va="center", fontsize=7.5, color=TAG_F,
        fontfamily="monospace", multialignment="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="#E8F8F5", ec=TAG_F, lw=1.2),
        zorder=4, linespacing=1.45)

# Arrow between the two Stage-1 boxes
arr(P4CX - 0.28, 1.82, P4CX + 0.38, 1.82,
    color=SEG_F, lw=2.0, hw=0.22, hl=0.28,
    label="compress", lside="right")

ax.text(P4CX, PY1 + 0.38,
        "Loss computed on target tokens only  (prompt masked with −100)",
        ha="center", va="center", fontsize=7.5, color=MID, style="italic", zorder=5)


# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG, pad_inches=0.15)
print(f"Saved → {OUT}")
