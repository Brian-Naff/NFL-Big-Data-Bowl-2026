"""
NFL Big Data Bowl 2026 — Player Movement Prediction Inference Pipeline

"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import pickle

class Config:
 
    MODEL_DIR = Path("/kaggle/input/nfl-exact-058-models/outputs")
    WINDOW_SIZE = 10  # number of observed frames passed to the sequence model
    MAX_FUTURE_HORIZON = 94
    K_NEIGH = 6       # maximum number of nearby players used in neighbor features
    RADIUS = 30.0     # yards; ignore neighbors farther away than this radius
    TAU = 8.0         # distance-decay scale for neighbor weighting
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_models = None
_scalers = None
_route_kmeans = None
_route_scaler = None

def load_models_once():
    """Load saved model folds and preprocessing artifacts once per session.

    Kaggle calls "predict" repeatedly through the inference server, so caching
    these artifacts avoids reloading model weights and scalers on every batch.
    """
    global _models, _scalers, _route_kmeans, _route_scaler
    if _models is not None:
        return
    
    with open(Config.MODEL_DIR / 'route_artifacts.pkl', 'rb') as f:
        route_artifacts = pickle.load(f)
    _route_kmeans = route_artifacts['kmeans']
    _route_scaler = route_artifacts['scaler']
    
    _models = []
    _scalers = []
    for fold in range(1, 6):
        checkpoint = torch.load(Config.MODEL_DIR / f'exact_058_model_fold{fold}.pth', map_location=Config.DEVICE)
        model = SeqModel(checkpoint['input_dim'], checkpoint['horizon']).to(Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        _models.append(model)
        
        with open(Config.MODEL_DIR / f'scaler_fold{fold}.pkl', 'rb') as f:
            _scalers.append(pickle.load(f))

class SeqModel(nn.Module):
    """GRU-attention trajectory model.

        Predicts futre displacements
    """
    def __init__(self, input_dim, horizon):
        super().__init__()
        self.gru = nn.GRU(input_dim, 128, num_layers=2, batch_first=True, dropout=0.1, bidirectional=False)
        self.pool_ln = nn.LayerNorm(128)
        self.pool_attn = nn.MultiheadAttention(128, num_heads=4, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, 128))
        self.head = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, horizon * 2)
        )
    
    def forward(self, x):
        h, _ = self.gru(x)
        B = h.size(0)
        q = self.pool_query.expand(B, -1, -1)
        ctx, _ = self.pool_attn(q, self.pool_ln(h), self.pool_ln(h))
        out = self.head(ctx.squeeze(1))
        return torch.cumsum(out.view(B, -1, 2), dim=1)

def get_velocity(speed, direction_deg):
    """Convert speed and direction angle into x/y velocity components."""
    theta = np.deg2rad(direction_deg)
    return speed * np.sin(theta), speed * np.cos(theta)

def height_to_feet(height_str):
    """Convert NFL height strings such as '6-2' into decimal feet."""
    try:
        ft, inches = map(int, str(height_str).split('-'))
        return ft + inches/12
    except:
        return 6.0

def compute_geometric_endpoint(df):
    """Create a role-aware geometric endpoint estimate for each player.

    Features:
    - targeted receivers are pulled toward the ball landing point,
    - coverage defenders are pulled toward a mirrored receiver endpoint when a
      plausible receiver match exists,
    - other players default to a short constant-velocity projection.

    """
    df = df.copy()
    t_total = df.get('num_frames_output', 30) / 10.0
    df['time_to_endpoint'] = t_total
    df['geo_endpoint_x'] = df['x'] + df['velocity_x'] * t_total
    df['geo_endpoint_y'] = df['y'] + df['velocity_y'] * t_total
    
    if 'ball_land_x' in df.columns:
        receiver_mask = df['player_role'] == 'Targeted Receiver'
        df.loc[receiver_mask, 'geo_endpoint_x'] = df.loc[receiver_mask, 'ball_land_x']
        df.loc[receiver_mask, 'geo_endpoint_y'] = df.loc[receiver_mask, 'ball_land_y']
        
        defender_mask = df['player_role'] == 'Defensive Coverage'
        has_mirror = df.get('mirror_offset_x', 0).notna() & (df.get('mirror_wr_dist', 50) < 15)
        coverage_mask = defender_mask & has_mirror
        
        df.loc[coverage_mask, 'geo_endpoint_x'] = (df.loc[coverage_mask, 'ball_land_x'] + df.loc[coverage_mask, 'mirror_offset_x'].fillna(0))
        df.loc[coverage_mask, 'geo_endpoint_y'] = (df.loc[coverage_mask, 'ball_land_y'] + df.loc[coverage_mask, 'mirror_offset_y'].fillna(0))
    
    near_left = df['y'] < 5
    near_right = df['y'] > 48.3
    df.loc[near_left & (df['geo_endpoint_y'] < 0), 'geo_endpoint_y'] = 2.0
    df.loc[near_right & (df['geo_endpoint_y'] > 53.3), 'geo_endpoint_y'] = 51.3
    
    df['geo_endpoint_x'] = df['geo_endpoint_x'].clip(0, 120)
    df['geo_endpoint_y'] = df['geo_endpoint_y'].clip(0, 53.3)
    return df

def add_geometric_features(df):
    """Add features describing how hard it is to reach the geometric endpoint.

    These features summarize distance, required velocity/acceleration, alignment
    with current motion, sideline constraints, turning angle.
    """
    df = compute_geometric_endpoint(df)
    df['geo_vector_x'] = df['geo_endpoint_x'] - df['x']
    df['geo_vector_y'] = df['geo_endpoint_y'] - df['y']
    df['geo_distance'] = np.sqrt(df['geo_vector_x']**2 + df['geo_vector_y']**2)
    
    t = df['time_to_endpoint'] + 0.1
    df['geo_required_vx'] = df['geo_vector_x'] / t
    df['geo_required_vy'] = df['geo_vector_y'] / t
    df['geo_velocity_error_x'] = df['geo_required_vx'] - df['velocity_x']
    df['geo_velocity_error_y'] = df['geo_required_vy'] - df['velocity_y']
    df['geo_velocity_error'] = np.sqrt(df['geo_velocity_error_x']**2 + df['geo_velocity_error_y']**2)
    
    t_sq = t * t
    df['geo_required_ax'] = (2 * df['geo_vector_x'] / t_sq).clip(-10, 10)
    df['geo_required_ay'] = (2 * df['geo_vector_y'] / t_sq).clip(-10, 10)
    
    velocity_mag = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    geo_unit_x = df['geo_vector_x'] / (df['geo_distance'] + 0.1)
    geo_unit_y = df['geo_vector_y'] / (df['geo_distance'] + 0.1)
    df['geo_alignment'] = (df['velocity_x'] * geo_unit_x + df['velocity_y'] * geo_unit_y) / (velocity_mag + 0.1)
    
    df['geo_dist_to_left_sideline'] = df['geo_endpoint_y']
    df['geo_dist_to_right_sideline'] = 53.3 - df['geo_endpoint_y']
    df['geo_near_sideline'] = ((df['geo_dist_to_left_sideline'] < 5) | (df['geo_dist_to_right_sideline'] < 5)).astype(float)
    
    current_angle = np.arctan2(df['velocity_y'], df['velocity_x'])
    required_angle = np.arctan2(df['geo_vector_y'], df['geo_vector_x'])
    angle_diff = required_angle - current_angle
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
    df['geo_required_turn'] = np.abs(angle_diff)
    
    max_accel = 8.0
    reachable_dist = df['s'] * t + 0.5 * max_accel * t_sq
    df['geo_is_reachable'] = (df['geo_distance'] <= reachable_dist).astype(float)
    df['geo_urgency'] = df['geo_distance'] / (reachable_dist + 0.1)
    
    endpoint_speed = np.sqrt(df['geo_required_vx']**2 + df['geo_required_vy']**2)
    df['geo_decel_needed'] = np.maximum(0, df['s'] - endpoint_speed) / (t + 0.1)
    
    return df

def get_opponent_features(input_df):
    """Summarize immediate opponent pressure at the last observed frame.

    For each player, this computes nearest-opponent distance, nearby opponent
    counts, closing speed, and coverage-defender mirroring features. These are
    intended to give the model local defensive context.
    """
    features = []
    for (gid, pid), group in input_df.groupby(['game_id', 'play_id']):
        last = group.sort_values('frame_id').groupby('nfl_id').last()
        if len(last) < 2:
            continue
        
        positions = last[['x', 'y']].values
        sides = last['player_side'].values
        speeds = last['s'].values
        directions = last['dir'].values
        roles = last['player_role'].values
        receiver_mask = np.isin(roles, ['Targeted Receiver', 'Other Route Runner'])
        
        for i, (nid, side, role) in enumerate(zip(last.index, sides, roles)):
            opp_mask = sides != side
            feat = {'game_id': gid, 'play_id': pid, 'nfl_id': nid, 'nearest_opp_dist': 50.0, 'closing_speed': 0.0,
                    'num_nearby_opp_3': 0, 'num_nearby_opp_5': 0, 'mirror_wr_vx': 0.0, 'mirror_wr_vy': 0.0,
                    'mirror_offset_x': 0.0, 'mirror_offset_y': 0.0, 'mirror_wr_dist': 50.0}
            
            if not opp_mask.any():
                features.append(feat)
                continue
            
            opp_positions = positions[opp_mask]
            distances = np.sqrt(((positions[i] - opp_positions)**2).sum(axis=1))
            if len(distances) == 0:
                features.append(feat)
                continue
            
            nearest_idx = distances.argmin()
            feat['nearest_opp_dist'] = distances[nearest_idx]
            feat['num_nearby_opp_3'] = (distances < 3.0).sum()
            feat['num_nearby_opp_5'] = (distances < 5.0).sum()
            
            my_vx, my_vy = get_velocity(speeds[i], directions[i])
            opp_vx, opp_vy = get_velocity(speeds[opp_mask][nearest_idx], directions[opp_mask][nearest_idx])
            rel_vx, rel_vy = my_vx - opp_vx, my_vy - opp_vy
            to_me = positions[i] - opp_positions[nearest_idx]
            to_me_norm = to_me / (np.linalg.norm(to_me) + 0.1)
            feat['closing_speed'] = -(rel_vx * to_me_norm[0] + rel_vy * to_me_norm[1])
            
            if role == 'Defensive Coverage' and receiver_mask.any():
                rec_positions = positions[receiver_mask]
                rec_distances = np.sqrt(((positions[i] - rec_positions)**2).sum(axis=1))
                if len(rec_distances) > 0:
                    closest_rec_idx = rec_distances.argmin()
                    rec_indices = np.where(receiver_mask)[0]
                    actual_rec_idx = rec_indices[closest_rec_idx]
                    rec_vx, rec_vy = get_velocity(speeds[actual_rec_idx], directions[actual_rec_idx])
                    feat['mirror_wr_vx'] = rec_vx
                    feat['mirror_wr_vy'] = rec_vy
                    feat['mirror_wr_dist'] = rec_distances[closest_rec_idx]
                    feat['mirror_offset_x'] = positions[i][0] - rec_positions[closest_rec_idx][0]
                    feat['mirror_offset_y'] = positions[i][1] - rec_positions[closest_rec_idx][1]
            
            features.append(feat)
    return pd.DataFrame(features)

def extract_route_patterns(input_df):
    """Extract simple recent-trajectory shape features.
    """
    route_features = []
    for (gid, pid, nid), group in input_df.groupby(['game_id', 'play_id', 'nfl_id']):
        traj = group.sort_values('frame_id').tail(5)
        if len(traj) < 3:
            continue
        
        positions = traj[['x', 'y']].values
        speeds = traj['s'].values
        
        total_dist = np.sum(np.sqrt(np.diff(positions[:, 0])**2 + np.diff(positions[:, 1])**2))
        displacement = np.sqrt((positions[-1, 0] - positions[0, 0])**2 + (positions[-1, 1] - positions[0, 1])**2)
        straightness = displacement / (total_dist + 0.1)
        
        angles = np.arctan2(np.diff(positions[:, 1]), np.diff(positions[:, 0]))
        if len(angles) > 1:
            angle_changes = np.abs(np.diff(angles))
            max_turn, mean_turn = np.max(angle_changes), np.mean(angle_changes)
        else:
            max_turn = mean_turn = 0
        
        speed_mean = speeds.mean()
        speed_change = speeds[-1] - speeds[0] if len(speeds) > 1 else 0
        dx, dy = positions[-1, 0] - positions[0, 0], positions[-1, 1] - positions[0, 1]
        
        route_features.append({
            'game_id': gid, 'play_id': pid, 'nfl_id': nid, 'traj_straightness': straightness,
            'traj_max_turn': max_turn, 'traj_mean_turn': mean_turn, 'traj_depth': abs(dx),
            'traj_width': abs(dy), 'speed_mean': speed_mean, 'speed_change': speed_change
        })
    
    route_df = pd.DataFrame(route_features)
    feat_cols = ['traj_straightness', 'traj_max_turn', 'traj_mean_turn', 'traj_depth', 'traj_width', 'speed_mean', 'speed_change']
    X = route_df[feat_cols].fillna(0)
    X_scaled = _route_scaler.transform(X)
    route_df['route_pattern'] = _route_kmeans.predict(X_scaled)
    return route_df

def compute_neighbor_embeddings(input_df):
    """Build graph-style local context features from nearby players.

    For each player, nearby teammates and opponents are summarized with
    distance-weighted relative position and velocity. 
    """
    cols_needed = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y", "velocity_x", "velocity_y", "player_side"]
    src = input_df[cols_needed].copy()
    
    last = (src.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
               .groupby(["game_id", "play_id", "nfl_id"], as_index=False).tail(1)
               .rename(columns={"frame_id": "last_frame_id"}).reset_index(drop=True))
    
    tmp = last.merge(src.rename(columns={"frame_id": "nb_frame_id", "nfl_id": "nfl_id_nb", "x": "x_nb", "y": "y_nb",
                                         "velocity_x": "vx_nb", "velocity_y": "vy_nb", "player_side": "player_side_nb"}),
                     left_on=["game_id", "play_id", "last_frame_id"],
                     right_on=["game_id", "play_id", "nb_frame_id"], how="left")
    
    tmp = tmp[tmp["nfl_id_nb"] != tmp["nfl_id"]]
    tmp["dx"], tmp["dy"] = tmp["x_nb"] - tmp["x"], tmp["y_nb"] - tmp["y"]
    tmp["dvx"], tmp["dvy"] = tmp["vx_nb"] - tmp["velocity_x"], tmp["vy_nb"] - tmp["velocity_y"]
    tmp["dist"] = np.sqrt(tmp["dx"]**2 + tmp["dy"]**2)
    tmp = tmp[np.isfinite(tmp["dist"]) & (tmp["dist"] > 1e-6)]
    
    if Config.RADIUS:
        tmp = tmp[tmp["dist"] <= Config.RADIUS]
    
    tmp["is_ally"] = (tmp["player_side_nb"] == tmp["player_side"]).astype(np.float32)
    keys = ["game_id", "play_id", "nfl_id"]
    tmp["rnk"] = tmp.groupby(keys)["dist"].rank(method="first")
    
    if Config.K_NEIGH:
        tmp = tmp[tmp["rnk"] <= float(Config.K_NEIGH)]
    
    tmp["w"] = np.exp(-tmp["dist"] / float(Config.TAU))
    sum_w = tmp.groupby(keys)["w"].transform("sum")
    tmp["wn"] = np.where(sum_w > 0, tmp["w"] / sum_w, 0.0)
    tmp["wn_ally"], tmp["wn_opp"] = tmp["wn"] * tmp["is_ally"], tmp["wn"] * (1.0 - tmp["is_ally"])
    
    for col in ["dx", "dy", "dvx", "dvy"]:
        tmp[f"{col}_ally_w"] = tmp[col] * tmp["wn_ally"]
        tmp[f"{col}_opp_w"] = tmp[col] * tmp["wn_opp"]
    
    tmp["dist_ally"] = np.where(tmp["is_ally"] > 0.5, tmp["dist"], np.nan)
    tmp["dist_opp"] = np.where(tmp["is_ally"] < 0.5, tmp["dist"], np.nan)
    
    ag = tmp.groupby(keys).agg(
        gnn_ally_dx_mean=("dx_ally_w", "sum"), gnn_ally_dy_mean=("dy_ally_w", "sum"),
        gnn_ally_dvx_mean=("dvx_ally_w", "sum"), gnn_ally_dvy_mean=("dvy_ally_w", "sum"),
        gnn_opp_dx_mean=("dx_opp_w", "sum"), gnn_opp_dy_mean=("dy_opp_w", "sum"),
        gnn_opp_dvx_mean=("dvx_opp_w", "sum"), gnn_opp_dvy_mean=("dvy_opp_w", "sum"),
        gnn_ally_cnt=("is_ally", "sum"), gnn_opp_cnt=("is_ally", lambda s: float(len(s) - s.sum())),
        gnn_ally_dmin=("dist_ally", "min"), gnn_ally_dmean=("dist_ally", "mean"),
        gnn_opp_dmin=("dist_opp", "min"), gnn_opp_dmean=("dist_opp", "mean")
    ).reset_index()
    
    near = tmp.loc[tmp["rnk"] <= 3, keys + ["rnk", "dist"]].copy()
    if len(near) > 0:
        near["rnk"] = near["rnk"].astype(int)
        dwide = near.pivot_table(index=keys, columns="rnk", values="dist", aggfunc="first")
        dwide = dwide.rename(columns={1: "gnn_d1", 2: "gnn_d2", 3: "gnn_d3"}).reset_index()
        ag = ag.merge(dwide, on=keys, how="left")
    
    for c in ["gnn_ally_dx_mean", "gnn_ally_dy_mean", "gnn_ally_dvx_mean", "gnn_ally_dvy_mean",
              "gnn_opp_dx_mean", "gnn_opp_dy_mean", "gnn_opp_dvx_mean", "gnn_opp_dvy_mean"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_cnt", "gnn_opp_cnt"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_dmin", "gnn_opp_dmin", "gnn_ally_dmean", "gnn_opp_dmean", "gnn_d1", "gnn_d2", "gnn_d3"]:
        ag[c] = ag[c].fillna(Config.RADIUS if Config.RADIUS else 30.0)
    
    return ag

def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pd.DataFrame:
    
    load_models_once()
    
    test_df = test.to_pandas()
    input_df = test_input.to_pandas()
    input_df = input_df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    
    # ------------------------------------------------------------------
    # Player body-size features
    # ------------------------------------------------------------------
    # These features are simple proxies for player physical profile. They are
    # kept in inference because the saved scalers and model were trained with
    # the same feature set.
    input_df['player_height_feet'] = input_df['player_height'].apply(height_to_feet)
    height_parts = input_df['player_height'].str.split('-', expand=True)
    input_df['height_inches'] = height_parts[0].astype(float) * 12 + height_parts[1].astype(float)
    input_df['bmi'] = (input_df['player_weight'] / (input_df['height_inches']**2)) * 703
    
    # ------------------------------------------------------------------
    # Kinematic features
    # ------------------------------------------------------------------
    # Convert tracking speed/direction into velocity, acceleration, momentum,
    # and orientation features. The exact formulas must match training-time
    # preprocessing so the saved scalers and model weights remain valid.
    dir_rad = np.deg2rad(input_df['dir'].fillna(0))
    input_df['velocity_x'] = input_df['s'] * np.sin(dir_rad)
    input_df['velocity_y'] = input_df['s'] * np.cos(dir_rad)
    input_df['acceleration_x'] = input_df['a'] * np.cos(dir_rad)
    input_df['acceleration_y'] = input_df['a'] * np.sin(dir_rad)
    input_df['speed_squared'] = input_df['s'] ** 2
    input_df['accel_magnitude'] = np.sqrt(input_df['acceleration_x']**2 + input_df['acceleration_y']**2)
    input_df['momentum_x'], input_df['momentum_y'] = input_df['velocity_x'] * input_df['player_weight'], input_df['velocity_y'] * input_df['player_weight']
    input_df['kinetic_energy'] = 0.5 * input_df['player_weight'] * input_df['speed_squared']
    input_df['orientation_diff'] = np.minimum(np.abs(input_df['o'] - input_df['dir']), 360 - np.abs(input_df['o'] - input_df['dir']))
    
    # ------------------------------------------------------------------
    # Role and side indicators
    # ------------------------------------------------------------------
    # Player role is highly predictive in this task: a targeted receiver,
    # coverage defender, and passer have very different movement objectives.
    input_df['is_offense'] = (input_df['player_side'] == 'Offense').astype(int)
    input_df['is_defense'] = (input_df['player_side'] == 'Defense').astype(int)
    input_df['is_receiver'] = (input_df['player_role'] == 'Targeted Receiver').astype(int)
    input_df['is_coverage'] = (input_df['player_role'] == 'Defensive Coverage').astype(int)
    input_df['is_passer'] = (input_df['player_role'] == 'Passer').astype(int)
    input_df['role_targeted_receiver'], input_df['role_defensive_coverage'], input_df['role_passer'] = input_df['is_receiver'], input_df['is_coverage'], input_df['is_passer']
    input_df['side_offense'] = input_df['is_offense']
    
    # ------------------------------------------------------------------
    # Ball landing context
    # ------------------------------------------------------------------
    # The Big Data Bowl task provides ball landing location at inference time.
    # These features describe each player's distance, direction, and velocity
    # alignment relative to that landing point.
    if 'ball_land_x' in input_df.columns:
        ball_dx, ball_dy = input_df['ball_land_x'] - input_df['x'], input_df['ball_land_y'] - input_df['y']
        input_df['distance_to_ball'] = np.sqrt(ball_dx**2 + ball_dy**2)
        input_df['dist_to_ball'], input_df['dist_squared'] = input_df['distance_to_ball'], input_df['distance_to_ball'] ** 2
        input_df['angle_to_ball'] = np.arctan2(ball_dy, ball_dx)
        input_df['ball_direction_x'], input_df['ball_direction_y'] = ball_dx / (input_df['distance_to_ball'] + 1e-6), ball_dy / (input_df['distance_to_ball'] + 1e-6)
        input_df['closing_speed_ball'] = input_df['velocity_x'] * input_df['ball_direction_x'] + input_df['velocity_y'] * input_df['ball_direction_y']
        input_df['velocity_toward_ball'] = input_df['velocity_x'] * np.cos(input_df['angle_to_ball']) + input_df['velocity_y'] * np.sin(input_df['angle_to_ball'])
        input_df['velocity_alignment'] = np.cos(input_df['angle_to_ball'] - dir_rad)
        input_df['angle_diff'] = np.minimum(np.abs(input_df['o'] - np.degrees(input_df['angle_to_ball'])), 360 - np.abs(input_df['o'] - np.degrees(input_df['angle_to_ball'])))
    
    # ------------------------------------------------------------------
    # Local spatial context features
    # ------------------------------------------------------------------
    # These three feature groups summarize opponent pressure, route shape, and
    # nearby-player context before the temporal sequence is passed to the model.
    opp_features = get_opponent_features(input_df)
    input_df = input_df.merge(opp_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    route_features = extract_route_patterns(input_df)
    input_df = input_df.merge(route_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    gnn_features = compute_neighbor_embeddings(input_df)
    input_df = input_df.merge(gnn_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    
    # Convert nearest-opponent distance into simple pressure indicators.
    # Higher pressure means less local space and usually stronger path
    # constraints on the player's future movement.
    if 'nearest_opp_dist' in input_df.columns:
        input_df['pressure'] = 1 / np.maximum(input_df['nearest_opp_dist'], 0.5)
        input_df['under_pressure'] = (input_df['nearest_opp_dist'] < 3).astype(int)
        input_df['pressure_x_speed'] = input_df['pressure'] * input_df['s']
    
    # For coverage defenders, compare their movement to the nearest receiver.
    # This approximates man-coverage mirroring and separation behavior.
    if 'mirror_wr_vx' in input_df.columns:
        s_safe = np.maximum(input_df['s'], 0.1)
        input_df['mirror_similarity'] = (input_df['velocity_x'] * input_df['mirror_wr_vx'] + input_df['velocity_y'] * input_df['mirror_wr_vy']) / s_safe
        input_df['mirror_offset_dist'] = np.sqrt(input_df['mirror_offset_x']**2 + input_df['mirror_offset_y']**2)
        input_df['mirror_alignment'] = input_df['mirror_similarity'] * input_df['role_defensive_coverage']
    
    # ------------------------------------------------------------------
    # Temporal history features
    # ------------------------------------------------------------------
    # Lags, rolling statistics, deltas, and EMAs give the model recent motion
    # history beyond the raw sequence window.
    gcols = ['game_id', 'play_id', 'nfl_id']
    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            if col in input_df.columns:
                input_df[f'{col}_lag{lag}'] = input_df.groupby(gcols)[col].shift(lag)
    
    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            if col in input_df.columns:
                input_df[f'{col}_rolling_mean_{window}'] = input_df.groupby(gcols)[col].rolling(window, min_periods=1).mean().reset_index(level=[0,1,2], drop=True)
                input_df[f'{col}_rolling_std_{window}'] = input_df.groupby(gcols)[col].rolling(window, min_periods=1).std().reset_index(level=[0,1,2], drop=True)
    
    for col in ['velocity_x', 'velocity_y']:
        if col in input_df.columns:
            input_df[f'{col}_delta'] = input_df.groupby(gcols)[col].diff()
    
    input_df['velocity_x_ema'] = input_df.groupby(gcols)['velocity_x'].transform(lambda x: x.ewm(alpha=0.3, adjust=False).mean())
    input_df['velocity_y_ema'] = input_df.groupby(gcols)['velocity_y'].transform(lambda x: x.ewm(alpha=0.3, adjust=False).mean())
    input_df['speed_ema'] = input_df.groupby(gcols)['s'].transform(lambda x: x.ewm(alpha=0.3, adjust=False).mean())
    
    # ------------------------------------------------------------------
    # Time-to-horizon features
    # ------------------------------------------------------------------
    # These features describe how much prediction horizon remains and how far a
    # simple constant-velocity projection would be from the ball landing point.
    if 'num_frames_output' in input_df.columns:
        max_frames = input_df['num_frames_output']
        input_df['max_play_duration'], input_df['frame_time'] = max_frames / 10.0, input_df['frame_id'] / 10.0
        input_df['progress_ratio'], input_df['time_remaining'] = input_df['frame_id'] / np.maximum(max_frames, 1), (max_frames - input_df['frame_id']) / 10.0
        input_df['frames_remaining'] = max_frames - input_df['frame_id']
        input_df['expected_x_at_ball'], input_df['expected_y_at_ball'] = input_df['x'] + input_df['velocity_x'] * input_df['frame_time'], input_df['y'] + input_df['velocity_y'] * input_df['frame_time']
        
        if 'ball_land_x' in input_df.columns:
            input_df['error_from_ball_x'], input_df['error_from_ball_y'] = input_df['expected_x_at_ball'] - input_df['ball_land_x'], input_df['expected_y_at_ball'] - input_df['ball_land_y']
            input_df['error_from_ball'] = np.sqrt(input_df['error_from_ball_x']**2 + input_df['error_from_ball_y']**2)
            input_df['weighted_dist_by_time'], input_df['dist_scaled_by_progress'] = input_df['dist_to_ball'] / (input_df['frame_time'] + 0.1), input_df['dist_to_ball'] * (1 - input_df['progress_ratio'])
        
        input_df['time_squared'] = input_df['frame_time'] ** 2
        input_df['velocity_x_progress'], input_df['velocity_y_progress'] = input_df['velocity_x'] * input_df['progress_ratio'], input_df['velocity_y'] * input_df['progress_ratio']
        input_df['speed_scaled_by_time_left'], input_df['actual_play_length'], input_df['length_ratio'] = input_df['s'] * input_df['time_remaining'], max_frames, max_frames / 30.0
    
    # Role-aware geometric endpoint features are added last because they use
    # several of the previously computed kinematic and mirroring features.
    input_df = add_geometric_features(input_df)

    # ------------------------------------------------------------------
    # Feature list used by the trained model
    # ------------------------------------------------------------------
    # The final set is filtered to columns present in the inference batch, but
    # the ordering must remain consistent with training and saved scalers.
    feature_cols = ['x', 'y', 's', 'a', 'o', 'dir', 'frame_id', 'ball_land_x', 'ball_land_y', 'player_height_feet', 'player_weight', 'height_inches', 'bmi',
                    'velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y', 'momentum_x', 'momentum_y', 'kinetic_energy', 'speed_squared', 'accel_magnitude', 'orientation_diff',
                    'is_offense', 'is_defense', 'is_receiver', 'is_coverage', 'is_passer', 'role_targeted_receiver', 'role_defensive_coverage', 'role_passer', 'side_offense',
                    'distance_to_ball', 'dist_to_ball', 'dist_squared', 'angle_to_ball', 'ball_direction_x', 'ball_direction_y', 'closing_speed_ball', 'velocity_toward_ball', 'velocity_alignment', 'angle_diff',
                    'nearest_opp_dist', 'closing_speed', 'num_nearby_opp_3', 'num_nearby_opp_5', 'mirror_wr_vx', 'mirror_wr_vy', 'mirror_offset_x', 'mirror_offset_y',
                    'pressure', 'under_pressure', 'pressure_x_speed', 'mirror_similarity', 'mirror_offset_dist', 'mirror_alignment',
                    'route_pattern', 'traj_straightness', 'traj_max_turn', 'traj_mean_turn', 'traj_depth', 'traj_width', 'speed_mean', 'speed_change',
                    'gnn_ally_dx_mean', 'gnn_ally_dy_mean', 'gnn_ally_dvx_mean', 'gnn_ally_dvy_mean', 'gnn_opp_dx_mean', 'gnn_opp_dy_mean', 'gnn_opp_dvx_mean', 'gnn_opp_dvy_mean',
                    'gnn_ally_cnt', 'gnn_opp_cnt', 'gnn_ally_dmin', 'gnn_ally_dmean', 'gnn_opp_dmin', 'gnn_opp_dmean', 'gnn_d1', 'gnn_d2', 'gnn_d3']
    
    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            feature_cols.append(f'{col}_lag{lag}')
    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            feature_cols.extend([f'{col}_rolling_mean_{window}', f'{col}_rolling_std_{window}'])
    feature_cols.extend(['velocity_x_delta', 'velocity_y_delta', 'velocity_x_ema', 'velocity_y_ema', 'speed_ema',
                        'max_play_duration', 'frame_time', 'progress_ratio', 'time_remaining', 'frames_remaining', 'expected_x_at_ball', 'expected_y_at_ball',
                        'error_from_ball_x', 'error_from_ball_y', 'error_from_ball', 'time_squared', 'weighted_dist_by_time', 'velocity_x_progress', 'velocity_y_progress',
                        'dist_scaled_by_progress', 'speed_scaled_by_time_left', 'actual_play_length', 'length_ratio',
                        'geo_endpoint_x', 'geo_endpoint_y', 'geo_vector_x', 'geo_vector_y', 'geo_distance', 'geo_required_vx', 'geo_required_vy',
                        'geo_velocity_error_x', 'geo_velocity_error_y', 'geo_velocity_error', 'geo_required_ax', 'geo_required_ay', 'geo_alignment',
                        'geo_dist_to_left_sideline', 'geo_dist_to_right_sideline', 'geo_near_sideline', 'geo_required_turn', 'geo_is_reachable', 'geo_urgency', 'geo_decel_needed'])
    
    feature_cols = [c for c in feature_cols if c in input_df.columns]
    
    # ------------------------------------------------------------------
    # Build fixed-length model sequences
    # ------------------------------------------------------------------
    # Each target player receives the last WINDOW_SIZE observed frames. Short
    # histories are front-padded and missing values are filled using that
    # player's available numeric history.
    input_df.set_index(['game_id', 'play_id', 'nfl_id'], inplace=True)
    grouped = input_df.groupby(level=['game_id', 'play_id', 'nfl_id'])
    target_groups = test_df[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
    
    sequences, sequence_ids = [], []
    for _, row in target_groups.iterrows():
        key = (row['game_id'], row['play_id'], row['nfl_id'])
        try:
            group_df = grouped.get_group(key)
        except KeyError:
            continue
        
        input_window = group_df.tail(Config.WINDOW_SIZE)
        if len(input_window) < Config.WINDOW_SIZE:
            pad_len = Config.WINDOW_SIZE - len(input_window)
            pad_df = pd.DataFrame(np.nan, index=range(pad_len), columns=input_window.columns)
            input_window = pd.concat([pad_df, input_window], ignore_index=True)
        
        input_window = input_window.fillna(group_df.mean(numeric_only=True))
        seq = input_window[feature_cols].values
        if np.isnan(seq).any():
            seq = np.nan_to_num(seq, nan=0.0)
        
        sequences.append(seq)
        sequence_ids.append(key)
    
    X_test = list(sequences)

    # The model predicts future displacement from the final observed position,
    # so the final x/y coordinate is added back after inference.
    x_last, y_last = np.array([s[-1, 0] for s in X_test]), np.array([s[-1, 1] for s in X_test])
    
    # ------------------------------------------------------------------
    # Five-fold ensemble inference
    # ------------------------------------------------------------------
    # Each fold has its own scaler because scaling was fit on that fold's
    # training data. Predictions are averaged for a more stable trajectory.
    all_preds = []
    for model, scaler in zip(_models, _scalers):
        X_sc = [scaler.transform(s) for s in X_test]
        X_t = torch.tensor(np.stack(X_sc).astype(np.float32)).to(Config.DEVICE)
        with torch.no_grad():
            preds = model(X_t).cpu().numpy()
        all_preds.append(preds)
    
    ens_preds = np.mean(all_preds, axis=0)
    H = ens_preds.shape[1]
    
    # ------------------------------------------------------------------
    # Format predictions for the Kaggle evaluation server
    # ------------------------------------------------------------------
    # Predictions are clipped to field bounds and returned in the same order as
    # the requested rows in `test`.
    result_rows = []
    for i, sid in enumerate(sequence_ids):
        player_test = test_df[(test_df['game_id'] == sid[0]) & (test_df['play_id'] == sid[1]) & (test_df['nfl_id'] == sid[2])].sort_values('frame_id')
        for t, _ in enumerate(player_test.itertuples()):
            tt = min(t, H - 1)
            px, py = np.clip(x_last[i] + ens_preds[i, tt, 0], 0, 120), np.clip(y_last[i] + ens_preds[i, tt, 1], 0, 53.3)
            result_rows.append({'x': float(px), 'y': float(py)})
    
    return pd.DataFrame(result_rows)
