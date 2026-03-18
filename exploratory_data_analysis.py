import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D

df_input  = pd.read_csv("train/input_2023_w01.csv")
df_output = pd.read_csv("train/output_2023_w01.csv")

PLAY_ID = 194

play_df       = df_input[df_input["play_id"] == PLAY_ID].copy()
output_play_df = df_output[df_output["play_id"] == PLAY_ID].copy()

play_df = play_df.merge(
    output_play_df[["play_id", "nfl_id"]].drop_duplicates(),
    on=["play_id", "nfl_id"], how="left"
)

print(f"Players found: {play_df['nfl_id'].nunique()}")

offense_df = play_df[play_df["player_side"] == "Offense"]
defense_df = play_df[play_df["player_side"] == "Defense"]

FIELD_X, FIELD_Y = 120, 53.3
xi = np.linspace(0, FIELD_X, 300)
yi = np.linspace(0, FIELD_Y, 150)
xx, yy = np.meshgrid(xi, yi)

fig, axes = plt.subplots(1, 2, figsize=(20, 7))
titles = ["Offense", "Defense"]
sides  = [offense_df, defense_df]
colors = ["cyan", "red"]

for ax, side_df, title, color in zip(axes, sides, titles, colors):
    ax.set_facecolor("#1a472a")
    ax.set_title(title, color="black", fontsize=13)  # ← was missing title per panel

    # Field lines
    for x in range(0, 121, 10):
        ax.axvline(x, color="white", lw=0.6, alpha=0.4)
    for x in np.arange(10, 110, 1):
        ax.plot([x, x], [18.5, 19.5], color="white", lw=0.5, alpha=0.4)
        ax.plot([x, x], [33.8, 34.8], color="white", lw=0.5, alpha=0.4)
    for ez in [(0, 10), (110, 120)]:
        ax.add_patch(patches.Rectangle((ez[0], 0), 10, FIELD_Y,
                     linewidth=0, facecolor="white", alpha=0.05))

    if len(side_df) > 1:
        # KDE heatmap
        values = np.vstack([side_df["x"], side_df["y"]])
        kernel = gaussian_kde(values, bw_method="scott")
        Z = kernel(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        ax.imshow(Z, origin="lower", extent=[0, FIELD_X, 0, FIELD_Y],
                  cmap="hot", alpha=0.6, aspect="auto")

        # Per-player paths
        for nfl_id, player in side_df.groupby("nfl_id"):
            name = player["player_name"].iloc[0]

            # Input path
            ax.plot(player["x"], player["y"], color=color, lw=1.2, alpha=0.7)
            ax.scatter(player["x"].iloc[0], player["y"].iloc[0],
                       color="lime", s=50, zorder=5)

            # Branch point (last known frame)
            branch_x = player["x"].iloc[-1]
            branch_y = player["y"].iloc[-1]
            ax.scatter(branch_x, branch_y,
                       color="yellow", s=70, zorder=6, marker="D")

            # Future path from output
            future = output_play_df[output_play_df["nfl_id"] == nfl_id]
            if len(future) > 0:
                future_x = [branch_x] + list(future["x"])
                future_y = [branch_y] + list(future["y"])
                ax.plot(future_x, future_y,
                        color="orange", lw=1.2, alpha=0.85,
                        linestyle="--", zorder=4)
                ax.scatter(future["x"].iloc[-1], future["y"].iloc[-1],
                           color="orange", s=50, zorder=5, marker="^")

            # Name label
            ax.text(player["x"].iloc[0], player["y"].iloc[0] + 0.8,
                    name.split()[-1],
                    color="white", fontsize=6, ha="center")

    ax.set_xlim(0, FIELD_X)
    ax.set_ylim(0, FIELD_Y)
    ax.set_xlabel("Field Length (yards)", color="black")
    ax.set_ylabel("Field Width (yards)", color="black")
    ax.tick_params(colors="black")

    legend_elements = [
        Line2D([0], [0], color="cyan",   lw=1.5, label="Offense path (input)"),
        Line2D([0], [0], color="red",    lw=1.5, label="Defense path (input)"),
        Line2D([0], [0], color="orange", lw=1.5, linestyle="--", label="Path while ball is in air (output)"),
        Line2D([0], [0], marker="o",  color="w", markerfacecolor="lime",   markersize=8, label="Start"),
        Line2D([0], [0], marker="D",  color="w", markerfacecolor="yellow", markersize=8, label="Branch point"),
        Line2D([0], [0], marker="^",  color="w", markerfacecolor="orange", markersize=8, label="Ball is caught/dropped"),
        Line2D([0], [0], marker="*",  color="w", markerfacecolor="white",  markersize=12, label="Ball landing"), 
    ]  # ← plt.scatter() inside a list was the main crash bug

    ax.legend(
        facecolor="#1a472a",
        handles=legend_elements,
        labelcolor="white",
        edgecolor="white",        
        loc="upper left",      
        fontsize=10,
        markerscale=1.2
    )
    
    ball_x = play_df["ball_land_x"].iloc[0]
    ball_y = play_df["ball_land_y"].iloc[0]
    ax.scatter(ball_x, ball_y, color="white", s=150, 
            zorder=10, marker="*", label="Ball landing")

plt.suptitle(f"Play {PLAY_ID} — All Players", color="black", fontsize=15, y=1.01)  
plt.tight_layout()
plt.savefig("heatmap_split.png", dpi=150, bbox_inches="tight")
plt.show()
