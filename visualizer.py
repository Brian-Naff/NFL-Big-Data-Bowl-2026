import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def to_pandas(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df.copy()


def plot_play_clean(
    pred_full,
    test_input,
    game_id=None,
    play_id=None,
    label_top_errors=3,
    show_observed=True,
):
    pred_full = to_pandas(pred_full)
    input_pd = to_pandas(test_input)

    if game_id is None or play_id is None:
        row = pred_full.iloc[0]
        game_id = row["game_id"]
        play_id = row["play_id"]

    play_pred = pred_full[
        (pred_full["game_id"] == game_id)
        & (pred_full["play_id"] == play_id)
    ].copy()

    play_input = input_pd[
        (input_pd["game_id"] == game_id)
        & (input_pd["play_id"] == play_id)
    ].copy()

    if play_pred.empty or play_input.empty:
        raise ValueError("No rows found for this game/play.")

    has_actual = {"actual_x", "actual_y"}.issubset(play_pred.columns)

    if has_actual:
        endpoint_errors = (
            play_pred.dropna(subset=["actual_x", "actual_y"])
            .sort_values("frame_id")
            .groupby("nfl_id", as_index=False)
            .tail(1)
            .copy()
        )

        endpoint_errors["endpoint_error"] = np.sqrt(
            (endpoint_errors["pred_x"] - endpoint_errors["actual_x"]) ** 2
            + (endpoint_errors["pred_y"] - endpoint_errors["actual_y"]) ** 2
        )

        top_error_ids = set(
            endpoint_errors
            .sort_values("endpoint_error", ascending=False)
            .head(label_top_errors)["nfl_id"]
        )
    else:
        top_error_ids = set()

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Actual vs. Predicted Future Paths — Game {game_id}, Play {play_id}")

    for x in range(10, 120, 10):
        ax.axvline(x, linestyle="--", alpha=0.15)

    if "absolute_yardline_number" in play_input.columns:
        los = play_input["absolute_yardline_number"].iloc[0]
        ax.axvline(
            los,
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Line of Scrimmage",
        )

    if "ball_land_x" in play_input.columns:
        ax.scatter(
            play_input["ball_land_x"].iloc[0],
            play_input["ball_land_y"].iloc[0],
            marker="*",
            s=220,
            label="Ball landing point",
            zorder=8,
        )

    actual_color = "green"
    pred_color = "orange"

    for nfl_id in play_pred["nfl_id"].unique():
        obs = play_input[play_input["nfl_id"] == nfl_id].sort_values("frame_id")
        pred = play_pred[play_pred["nfl_id"] == nfl_id].sort_values("frame_id")

        if obs.empty or pred.empty:
            continue

        side = str(obs["player_side"].iloc[-1]) if "player_side" in obs.columns else ""
        role = str(obs["player_role"].iloc[-1]) if "player_role" in obs.columns else ""
        pos = str(obs["player_position"].iloc[-1]) if "player_position" in obs.columns else ""

        is_offense = "Offense" in side
        marker_color = "blue" if is_offense else "red"

        is_key_player = role in ["Targeted Receiver", "Passer"] or nfl_id in top_error_ids

        line_width = 1.6 if is_key_player else 0.8
        line_alpha = 0.9 if is_key_player else 0.35

        if show_observed:
            ax.plot(
                obs["x"],
                obs["y"],
                color="gray",
                linewidth=0.8,
                alpha=0.30,
            )

        if has_actual and pred["actual_x"].notna().any():
            ax.plot(
                pred["actual_x"],
                pred["actual_y"],
                color=actual_color,
                linewidth=line_width,
                alpha=line_alpha,
            )

        ax.plot(
            pred["pred_x"],
            pred["pred_y"],
            color=pred_color,
            linestyle="--",
            linewidth=line_width,
            alpha=line_alpha,
        )

        if has_actual and pred["actual_x"].notna().any():
            actual_end = pred.dropna(subset=["actual_x", "actual_y"]).iloc[-1]
            pred_end = pred.iloc[-1]

            ax.plot(
                [actual_end["actual_x"], pred_end["pred_x"]],
                [actual_end["actual_y"], pred_end["pred_y"]],
                color="black",
                linestyle=":",
                linewidth=0.7,
                alpha=0.6,
            )

        ax.scatter(
            obs["x"].iloc[-1],
            obs["y"].iloc[-1],
            color=marker_color,
            s=60 if is_key_player else 30,
            edgecolors="white",
            linewidths=0.8,
            zorder=6,
        )

        if is_key_player:
            label = pos

            if role == "Targeted Receiver":
                label += " TR"
            elif role == "Passer":
                label += " QB"

            ax.text(
                obs["x"].iloc[-1],
                obs["y"].iloc[-1] + 0.7,
                label,
                fontsize=9,
                ha="center",
                va="bottom",
                color="black",
                zorder=9,
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    boxstyle="round,pad=0.2",
                    alpha=0.85,
                ),
            )

    ax.plot([], [], color="gray", linewidth=0.8, alpha=0.5, label="Observed input")
    ax.plot([], [], color=actual_color, linewidth=1.5, label="Actual future")
    ax.plot([], [], color=pred_color, linestyle="--", linewidth=1.5, label="Predicted future")
    ax.plot([], [], color="black", linestyle=":", linewidth=0.8, label="Endpoint error")

    ax.legend(loc="upper right")
    ax.grid(alpha=0.15)
    plt.show()
