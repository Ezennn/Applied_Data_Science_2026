import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Load data
df_input = pd.read_csv("train/input_2023_w01.csv")
df_output = pd.read_csv("train/output_2023_w01.csv")

PLAY_ID = 194
play_df = df_input[df_input["play_id"] == PLAY_ID].copy()
output_play_df = df_output[df_output["play_id"] == PLAY_ID].copy()

# Mark which players have predictions
predicted_ids = set(output_play_df["nfl_id"].unique())
play_df["is_predicted"] = play_df["nfl_id"].isin(predicted_ids)

# Tag output frames
output_play_df["frame_type"] = "predicted"
play_df["frame_type"]        = "input"

# Combine input + output into one df
output_play_df["player_name"] = output_play_df["nfl_id"].map(
    play_df[["nfl_id","player_name"]].drop_duplicates().set_index("nfl_id")["player_name"]
)
output_play_df["player_side"] = output_play_df["nfl_id"].map(
    play_df[["nfl_id","player_side"]].drop_duplicates().set_index("nfl_id")["player_side"]
)
output_play_df["player_role"] = output_play_df["nfl_id"].map(
    play_df[["nfl_id","player_role"]].drop_duplicates().set_index("nfl_id")["player_role"]
)
output_play_df["is_predicted"] = True

all_df = pd.concat([play_df, output_play_df], ignore_index=True)
all_df = all_df.sort_values(["nfl_id", "frame_id"])

# Global frame list (input frames only for slider)
input_frames = sorted(play_df["frame_id"].unique())
#All frames (input + output) for animation completeness
output_frames = sorted(output_play_df["frame_id"].unique())
all_frames    = sorted(set(input_frames) | set(output_frames))

BALL_X = play_df["ball_land_x"].iloc[0]
BALL_Y = play_df["ball_land_y"].iloc[0]

# ── Colour mapping ────────────────────────────────────────────
def get_color(row):
    if row["frame_type"] == "predicted":
        return "orange"
    elif row["player_side"] == "Offense":
        return "cyan"
    else:
        return "red"

# ── Build figure ──────────────────────────────────────────────
fig = go.Figure()

# ── Field background shape ────────────────────────────────────
def add_field_shapes(fig):
    # Grass
    fig.add_shape(type="rect", x0=0, x1=120, y0=0, y1=53.3,
                  fillcolor="#1a472a", line_color="#1a472a", layer="below")
    # Endzones
    for x0, x1 in [(0,10),(110,120)]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=53.3,
                      fillcolor="rgba(255,255,255,0.06)",
                      line_color="white", layer="below")
    # Yard lines
    for x in range(0, 121, 10):
        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=53.3,
                      line=dict(color="rgba(255,255,255,0.4)", width=1))
    # Hash marks
    for x in np.arange(10, 110, 1):
        for y0, y1 in [(18.5,19.5),(33.8,34.8)]:
            fig.add_shape(type="line", x0=x, x1=x, y0=y0, y1=y1,
                          line=dict(color="rgba(255,255,255,0.3)", width=0.5))
    # Yard number labels
    for yard in range(10, 110, 10):
        label = yard if yard <= 50 else 100 - yard
        fig.add_annotation(x=yard, y=2, text=str(label),
                           showarrow=False, font=dict(color="white", size=9))

add_field_shapes(fig)

# ── Add traces per player per frame type ─────────────────────
players = play_df["nfl_id"].unique()

for nfl_id in players:
    player_meta = play_df[play_df["nfl_id"] == nfl_id].iloc[0]
    name        = player_meta["player_name"]
    side        = player_meta["player_side"]
    is_pred     = nfl_id in predicted_ids

    input_data  = play_df[play_df["nfl_id"] == nfl_id].sort_values("frame_id")
    output_data = output_play_df[output_play_df["nfl_id"] == nfl_id].sort_values("frame_id")

    dot_color  = "cyan" if side == "Offense" else "red"
    path_color = dot_color

    # Input path line
    fig.add_trace(go.Scatter(
        x=input_data["x"], y=input_data["y"],
        mode="lines",
        line=dict(color=path_color, width=1.5, dash="solid"),
        name=f"{name} (input path)",
        legendgroup=str(nfl_id),
        showlegend=False,
        hoverinfo="skip",
        customdata=[[side, "input", str(nfl_id)]] * len(input_data),
        visible=True
    ))

    # Predicted path line
    if len(output_data) > 0:
        branch_x = input_data["x"].iloc[-1]
        branch_y = input_data["y"].iloc[-1]
        pred_x   = [branch_x] + list(output_data["x"])
        pred_y   = [branch_y] + list(output_data["y"])

        fig.add_trace(go.Scatter(
            x=pred_x, y=pred_y,
            mode="lines",
            line=dict(color="orange", width=1.5, dash="solid"),
            name=f"{name} (predicted)",
            legendgroup=str(nfl_id),  
            showlegend=False,
            hoverinfo="skip",
            visible=True
        ))

    # Player dot — animated per frame
    fig.add_trace(go.Scatter(
        x=[input_data["x"].iloc[0]],
        y=[input_data["y"].iloc[0]],
        mode="markers+text",
        marker=dict(size=12, color=dot_color,
                    symbol="circle",
                    line=dict(color="white", width=1.5)),
        text=[name.split()[-1]],
        textposition="top center",
        textfont=dict(color="white", size=8),
        name=name,
        legendgroup=str(nfl_id),
        showlegend=True,
        hovertemplate=(
            f"<b>{name}</b><br>"
            f"Side: {side}<br>"
            f"Predicted: {is_pred}<br>"
            "x: %{x:.1f}<br>y: %{y:.1f}<extra></extra>"
        ),
        customdata=[[side, "predicted" if is_pred else "not_predicted", str(nfl_id)]],
        visible=True
    ))

# Ball landing star
fig.add_trace(go.Scatter(
    x=[BALL_X], y=[BALL_Y],
    mode="markers",
    marker=dict(size=18, color="white", symbol="star"),
    name="Ball Landing",
    hovertemplate=f"Ball Landing<br>x: {BALL_X:.1f}<br>y: {BALL_Y:.1f}<extra></extra>"
))

# ── Animation frames ──────────────────────────────────────────
frames = []
for frame_id in all_frames:
    frame_data = []

    for nfl_id in players:
        input_data  = play_df[play_df["nfl_id"] == nfl_id].sort_values("frame_id")
        output_data = output_play_df[output_play_df["nfl_id"] == nfl_id].sort_values("frame_id")

        player_meta = play_df[play_df["nfl_id"] == nfl_id].iloc[0]
        name        = player_meta["player_name"]
        side        = player_meta["player_side"]
        is_pred     = nfl_id in predicted_ids
        dot_color   = "cyan" if side == "Offense" else "red"

        # Input path up to this frame
        path_so_far = input_data[input_data["frame_id"] <= frame_id]

        # Input path trace 
        frame_data.append(go.Scatter(
            x=path_so_far["x"], y=path_so_far["y"],
            mode="lines",
            line=dict(color=dot_color, width=1.5)
        ))

        # Predicted path (show only after last input frame)
        
        if len(output_data) > 0:
            if frame_id == input_frames[-1]:
                branch_x = input_data["x"].iloc[-1]
                branch_y = input_data["y"].iloc[-1]
                pred_x   = [branch_x] + list(output_data["x"])
                pred_y   = [branch_y] + list(output_data["y"])
            else:
                pred_x, pred_y = [], []
            frame_data.append(go.Scatter(
                x=pred_x, y=pred_y,
                mode="lines",
                line=dict(color="orange", width=1.5, dash="solid")
            ))

        # Player dot at current frame
        current = input_data[input_data["frame_id"] == frame_id]
        if len(current) == 0:
            current = input_data.iloc[[-1]]

        frame_data.append(go.Scatter(
            x=current["x"], y=current["y"],
            mode="markers+text",
            marker=dict(size=12, color=dot_color,
                        line=dict(color="white", width=1.5)),
            text=[name.split()[-1]],
            textposition="top center",
            textfont=dict(color="white", size=8)
        ))

    # Ball landing always visible
    frame_data.append(go.Scatter(
        x=[BALL_X], y=[BALL_Y],
        mode="markers",
        marker=dict(size=18, color="white", symbol="star")
    ))

    frames.append(go.Frame(data=frame_data, name=str(frame_id)))

fig.frames = frames

# ── Layout ────────────────────────────────────────────────────
fig.update_layout(
    title=dict(text=f"Play {PLAY_ID} — Frame by Frame",
               font=dict(color="white", size=16)),
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#1a472a",
    font=dict(color="white"),
    xaxis=dict(range=[0,120], showgrid=False, zeroline=False,
               title="Field Length (yards)", color="white"),
    yaxis=dict(range=[0,53.3], showgrid=False, zeroline=False,
               title="Field Width (yards)", color="white",
               scaleanchor="x", scaleratio=1),
    legend=dict(bgcolor="rgba(10,14,26,0.8)", bordercolor="white",
                borderwidth=1, font=dict(color="white")),
    updatemenus=[dict(
        type="buttons", showactive=False,
        y=1.05, x=0.1,
        buttons=[
            dict(label="▶ Play",
                 method="animate",
                 args=[None, dict(frame=dict(duration=100, redraw=True),
                                  fromcurrent=True)]),
            dict(label="⏸ Pause",
                 method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode="immediate")])
        ]
    )],
    sliders=[dict(
        steps=[dict(method="animate",
                    args=[[str(f)], dict(mode="immediate",
                                         frame=dict(duration=100, redraw=True))],
                    label=str(f)) for f in all_frames],
        x=0.1, y=0, len=0.9,
        currentvalue=dict(prefix="Frame: ", font=dict(color="white")),
        font=dict(color="white")
    )],
    height=650
)

# ── Filter buttons (Offense / Defense / Predicted) ────────────
fig.update_layout(
    updatemenus=[
        # Play/Pause
        dict(type="buttons", showactive=False, y=1.1, x=0.0, xanchor="left",
             buttons=[
                 dict(label="▶ Play", method="animate",
                      args=[None, dict(frame=dict(duration=100, redraw=True),
                                       fromcurrent=True)]),
                 dict(label="⏸ Pause", method="animate",
                      args=[[None], dict(frame=dict(duration=0, redraw=False),
                                         mode="immediate")])
             ]),
        # Side filter
        dict(type="buttons", showactive=True, y=1.1, x=0.25, xanchor="left",
             bgcolor="#1e2d4a", font=dict(color="white"),
             buttons=[
                 dict(label="All",     method="restyle",
                      args=[{"visible": True}]),
                 dict(label="Offense", method="restyle",
                      args=[{"visible": [
                          True if "Offense" in str(t.customdata) or
                          t.name == "Ball Landing" else "legendonly"
                          for t in fig.data
                      ]}]),
                 dict(label="Defense", method="restyle",
                      args=[{"visible": [
                          True if "Defense" in str(t.customdata) or
                          t.name == "Ball Landing" else "legendonly"
                          for t in fig.data
                      ]}]),
                 dict(label="Predicted Only", method="restyle",
                      args=[{"visible": [
                          True if (t.customdata is not None and any("predicted" == str(row[1]) for row in t.customdata)) or
                          t.name == "Ball Landing" else "legendonly"
                          for t in fig.data
                      ]}])
             ])
        
    ]
)

# ── Save as HTML dashboard ────────────────────────────────────
fig.write_html("play_dashboard.html")
print("✅ Dashboard saved: play_dashboard.html")
fig.show()
