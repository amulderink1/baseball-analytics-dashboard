"""
⚾ Baseball Analytics Dashboard v2.0
Unified interface for all baseball analytics reports

Reports included:
- Pitcher Trajectory Report (RHH/LHH splits, side view + catcher view with KDE zones)
- At-Bat Pitch Sequences (PDF)
- Team Offense Overview (spray chart)
- Hard-Hit Balls Report (CSV)
- Pitcher Scrimmage Report
- Hitter Scrimmage Report
- Foul Ball Zone Report (per-batter foul ball analysis in zone/shadow)

Run with: streamlit run main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Polygon, Wedge, Arc, Ellipse
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import ConvexHull
from scipy import stats
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import os
import io
import warnings
import glob
import re
# tkinter for local folder picker (not available on Streamlit Cloud)
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Baseball Analytics Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONSTANTS
# =============================================================================
PITCH_COLORS = {
    'Fastball': '#e74c3c',
    'Four-Seam': '#e74c3c',
    'Sinker': '#9b59b6',
    'Changeup': '#2ecc71',
    'ChangeUp': '#2ecc71',
    'Splitter': '#2ecc71',
    'Curveball': '#3498db',
    'Slider': '#f1c40f',
    'Cutter': '#FF8C00',
    'Sweeper': '#FF69B4',
    'Other': '#95a5a6'
}

MOUND_DISTANCE = 60.5
PLATE_Y = 1.417
PLATE_WIDTH = 17 / 12
STRIKE_ZONE_WIDTH = PLATE_WIDTH
STRIKE_ZONE_HEIGHT_LOW = 1.5
STRIKE_ZONE_HEIGHT_HIGH = 3.5
GRAVITY = 32.174

# Foul Ball Zone Report Constants
ZONE_LEFT = -0.83
ZONE_RIGHT = 0.83
ZONE_BOTTOM = 1.5  # matches STRIKE_ZONE_HEIGHT_LOW
ZONE_TOP = 3.5     # matches STRIKE_ZONE_HEIGHT_HIGH
SHADOW_BUFFER = 0.25

# UMBA Physics Model Constants
RHO_AIR = 0.074
BALL_DIAMETER = (2 + 15 / 16) / 12
BALL_AREA = 0.25 * np.pi * BALL_DIAMETER ** 2
BALL_MASS = 5.125 / 16
C0_AERO = 0.5 * RHO_AIR * BALL_AREA / BALL_MASS
CD_CONSTANT = 0.33


# =============================================================================
# DATA LOADING - WITH GOOGLE DRIVE FOLDER SUPPORT
# =============================================================================
def get_date_from_filename(filename):
    """Extract date from filename in YYYYMMDD format"""
    match = re.match(r'(\d{8})', os.path.basename(filename))
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d')
        except:
            return None
    return None


def load_csv_from_folder(folder_path, start_date=None, end_date=None):
    """Load CSV files from a folder, optionally filtered by date range"""
    all_data = []

    if not os.path.exists(folder_path):
        st.error(f"Folder not found: {folder_path}")
        return None

    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    for csv_file in csv_files:
        # Skip player positioning files
        if 'playerpositioning' in csv_file.lower():
            continue

        # Filter by date if specified
        file_date = get_date_from_filename(csv_file)
        if file_date:
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue

        try:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
            df['_source_file'] = os.path.basename(csv_file)
            all_data.append(df)
        except Exception as e:
            st.warning(f"Error loading {os.path.basename(csv_file)}: {e}")

    if not all_data:
        return None

    combined_df = pd.concat(all_data, ignore_index=True)

    # Standardize pitch types
    pitch_map = {
        'Four-Seam': 'Fastball',
        'FourSeamFastBall': 'Fastball',
        'Four-Seam Fastball': 'Fastball',
        'Two-Seam': 'Sinker',
        'TwoSeamFastBall': 'Sinker',
        'Two-Seam Fastball': 'Sinker',
        'ChangeUp': 'Changeup',
        'Change Up': 'Changeup'
    }
    if 'TaggedPitchType' in combined_df.columns:
        combined_df['TaggedPitchType'] = combined_df['TaggedPitchType'].replace(pitch_map)

    return combined_df


@st.cache_data
def load_csv_files(uploaded_files):
    """Load and combine uploaded CSV files"""
    all_data = []

    for uploaded_file in uploaded_files:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = df.columns.str.strip()
            # Skip player positioning files
            if 'playerpositioning' not in uploaded_file.name.lower():
                df['_source_file'] = uploaded_file.name
                all_data.append(df)
        except Exception as e:
            st.warning(f"Error loading {uploaded_file.name}: {e}")

    if not all_data:
        return None

    combined_df = pd.concat(all_data, ignore_index=True)

    # Standardize pitch types
    pitch_map = {
        'Four-Seam': 'Fastball',
        'FourSeamFastBall': 'Fastball',
        'Four-Seam Fastball': 'Fastball',
        'Two-Seam': 'Sinker',
        'TwoSeamFastBall': 'Sinker',
        'Two-Seam Fastball': 'Sinker',
        'ChangeUp': 'Changeup',
        'Change Up': 'Changeup'
    }
    if 'TaggedPitchType' in combined_df.columns:
        combined_df['TaggedPitchType'] = combined_df['TaggedPitchType'].replace(pitch_map)

    return combined_df


def get_data_summary(df):
    """Get summary stats from data"""
    summary = {
        'total_pitches': len(df),
        'dates': df['Date'].unique().tolist() if 'Date' in df.columns else [],
        'pitchers': sorted(df[df['PitcherTeam'] == 'SAN_BRO']['Pitcher'].dropna().unique().tolist()) if 'Pitcher' in df.columns else [],
        'batters': sorted(df[df['BatterTeam'] == 'SAN_BRO']['Batter'].dropna().unique().tolist()) if 'Batter' in df.columns else [],
        'teams': sorted(df['BatterTeam'].dropna().unique().tolist()) if 'BatterTeam' in df.columns else [],
        'balls_in_play': len(df[df['PitchCall'] == 'InPlay']) if 'PitchCall' in df.columns else 0
    }
    return summary


def get_pitch_color(pitch_type):
    """Get standardized color for pitch type"""
    return PITCH_COLORS.get(pitch_type, PITCH_COLORS['Other'])


# =============================================================================
# HITTING REPORTS - FIXED VERSION
# =============================================================================
def filter_quality_bip(df, team=None, min_ev=90, exclude_team=None):
    """Filter for quality balls in play - FIXED VERSION"""
    mask = (
        (df['ExitSpeed'].notna()) &
        (df['ExitSpeed'] >= min_ev) &
        (df['TaggedHitType'].notna()) &
        (df['TaggedHitType'] != 'Undefined') &
        (df['Direction'].notna())
    )

    if exclude_team:
        mask = mask & (df['BatterTeam'] != exclude_team)

    if team and team != exclude_team:
        mask = mask & (df['BatterTeam'] == team)

    bip = df[mask].copy()

    # Create EVCategory with dynamic bins based on min_ev
    # Handle cases where min_ev >= 95 or min_ev >= 100
    if min_ev >= 100:
        bip['EVCategory'] = 'High'
        bip['EVDisplay'] = '100+'
    elif min_ev >= 95:
        bip['EVCategory'] = pd.cut(
            bip['ExitSpeed'],
            bins=[min_ev, 100, float('inf')],
            labels=['Mid', 'High'],
            right=False
        )
        bip['EVDisplay'] = bip['ExitSpeed'].apply(
            lambda x: f'{min_ev}-100' if x < 100 else '100+'
        )
    else:
        bip['EVCategory'] = pd.cut(
            bip['ExitSpeed'],
            bins=[min_ev, 95, 100, float('inf')],
            labels=['Low', 'Mid', 'High'],
            right=False
        )
        bip['EVDisplay'] = bip['ExitSpeed'].apply(
            lambda x: f'{min_ev}-95' if x < 95 else ('95-100' if x < 100 else '100+')
        )

    return bip


def get_hit_color(hit_type, exit_speed, min_ev=90):
    """Get color based on hit type and exit speed - FIXED VERSION"""
    colors = {
        'GroundBall': {
            'Low': '#93C5FD',    # Light blue
            'Mid': '#3B82F6',    # Medium blue
            'High': '#1E40AF'    # Dark blue
        },
        'LineDrive': {
            'Low': '#86EFAC',    # Light green
            'Mid': '#22C55E',    # Medium green
            'High': '#15803D'    # Dark green
        },
        'FlyBall': {
            'Low': '#FCD34D',    # Light orange/yellow
            'Mid': '#F97316',    # Medium orange
            'High': '#DC2626'    # Red
        },
        'Popup': {
            'Low': '#E5E7EB',    # Light gray
            'Mid': '#9CA3AF',    # Medium gray
            'High': '#4B5563'    # Dark gray
        }
    }

    # Determine EV category based on speed
    if exit_speed < 95:
        ev_cat = 'Low'
    elif exit_speed < 100:
        ev_cat = 'Mid'
    else:
        ev_cat = 'High'

    return colors.get(hit_type, colors.get('FlyBall', {})).get(ev_cat, '#9CA3AF')


def convert_to_field_coords(bearing, distance):
    """Convert bearing and distance to field coordinates"""
    angle_rad = np.radians(bearing)
    max_distance = 400
    scale = min(distance / max_distance, 1)
    reach = 0.85 * scale
    x = 0.5 - reach * np.sin(angle_rad)
    y = 1 - reach * np.cos(angle_rad)
    return x, y


def draw_field(ax):
    """Draw baseball field outline"""
    field = patches.Wedge((0.5, 1), 0.85, -135, -45,
                          facecolor='#86efac', edgecolor='#22c55e',
                          linewidth=2, alpha=0.3)
    ax.add_patch(field)

    ax.plot([0.5, 0.05], [1, 0.15], color='#22c55e', linewidth=2, alpha=0.4)
    ax.plot([0.5, 0.95], [1, 0.15], color='#22c55e', linewidth=2, alpha=0.4)

    infield_points = np.array([
        [0.5, 1], [0.415, 0.83], [0.5, 0.75], [0.585, 0.83], [0.5, 1]
    ])
    infield = patches.Polygon(infield_points, facecolor='#d97706',
                              edgecolor='#92400e', alpha=0.6)
    ax.add_patch(infield)

    for x, y in [(0.415, 0.83), (0.5, 0.75), (0.585, 0.83)]:
        ax.plot(x, y, 'o', color='white', markersize=8,
                markeredgecolor='#15803d', markeredgewidth=2)

    ax.plot(0.5, 1, 's', color='white', markersize=10,
            markeredgecolor='#374151', markeredgewidth=2)
    ax.text(0.5, 1.02, 'HOME', ha='center', va='bottom', fontsize=10, fontweight='bold')


def create_team_spray_chart(bip_df, title="Team Offense Overview", min_ev=90):
    """Create team spray chart visualization - FIXED VERSION"""
    fig = plt.figure(figsize=(14, 12))

    dates = bip_df['Date'].unique() if 'Date' in bip_df.columns else []
    date_str = ', '.join(sorted([str(d) for d in dates]))

    total = len(bip_df)
    gb_count = len(bip_df[bip_df['TaggedHitType'] == 'GroundBall'])
    ld_count = len(bip_df[bip_df['TaggedHitType'] == 'LineDrive'])
    fb_count = len(bip_df[bip_df['TaggedHitType'] == 'FlyBall'])

    # Count by actual exit speed ranges - handles min_ev >= 95
    if min_ev >= 100:
        ev_low_count = 0
        ev_mid_count = 0
        ev_high_count = len(bip_df[bip_df['ExitSpeed'] >= 100])
    elif min_ev >= 95:
        ev_low_count = 0
        ev_mid_count = len(bip_df[(bip_df['ExitSpeed'] >= min_ev) & (bip_df['ExitSpeed'] < 100)])
        ev_high_count = len(bip_df[bip_df['ExitSpeed'] >= 100])
    else:
        ev_low_count = len(bip_df[(bip_df['ExitSpeed'] >= min_ev) & (bip_df['ExitSpeed'] < 95)])
        ev_mid_count = len(bip_df[(bip_df['ExitSpeed'] >= 95) & (bip_df['ExitSpeed'] < 100)])
        ev_high_count = len(bip_df[bip_df['ExitSpeed'] >= 100])

    fig.suptitle(title, fontsize=24, fontweight='bold', y=0.96)
    if date_str:
        fig.text(0.5, 0.91, f'Date(s): {date_str}', ha='center', fontsize=14, style='italic')

    ax_spray = fig.add_axes([0.1, 0.25, 0.8, 0.62])
    ax_spray.set_xlim(0, 1)
    ax_spray.set_ylim(0, 1.05)
    ax_spray.set_aspect('equal')
    ax_spray.axis('off')

    draw_field(ax_spray)

    for _, hit in bip_df.iterrows():
        bearing = hit.get('Bearing', hit.get('Direction', 0))
        distance = hit.get('Distance', 200)
        if pd.isna(bearing) or pd.isna(distance):
            continue
        x, y = convert_to_field_coords(bearing, distance)

        # Use the fixed color function with actual exit speed
        color = get_hit_color(hit['TaggedHitType'], hit['ExitSpeed'], min_ev)

        ax_spray.plot(x, y, 'o', color=color, markersize=16,
                      markeredgecolor='white', markeredgewidth=2.5, alpha=0.85)
        ax_spray.text(x, y - 0.02, f"{hit['ExitSpeed']:.1f}",
                      ha='center', va='top', fontsize=9, fontweight='bold')

    # Stats box - handles min_ev >= 95
    stats_text = f'Total: {total} | GB: {gb_count} | LD: {ld_count} | FB: {fb_count}\n'
    if min_ev >= 100:
        stats_text += f'100+: {ev_high_count}'
    elif min_ev >= 95:
        stats_text += f'{min_ev}-100: {ev_mid_count} | 100+: {ev_high_count}'
    else:
        stats_text += f'{min_ev}-95: {ev_low_count} | 95-100: {ev_mid_count} | 100+: {ev_high_count}'
    ax_spray.text(0.98, 0.98, stats_text, transform=ax_spray.transAxes,
                  ha='right', va='top', fontsize=12,
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Legend - Updated with dynamic labels based on min_ev
    ax_legend = fig.add_axes([0.1, 0.08, 0.8, 0.14])
    ax_legend.axis('off')

    if min_ev >= 100:
        legend_data = [
            ('Ground Ball', ['100+'], ['#1E40AF'], 0.05),
            ('Line Drive', ['100+'], ['#15803D'], 0.38),
            ('Fly Ball', ['100+'], ['#DC2626'], 0.71),
        ]
    elif min_ev >= 95:
        legend_data = [
            ('Ground Ball', [f'{min_ev}-100', '100+'], ['#3B82F6', '#1E40AF'], 0.05),
            ('Line Drive', [f'{min_ev}-100', '100+'], ['#22C55E', '#15803D'], 0.38),
            ('Fly Ball', [f'{min_ev}-100', '100+'], ['#F97316', '#DC2626'], 0.71),
        ]
    else:
        legend_data = [
            ('Ground Ball', [f'{min_ev}-95', '95-100', '100+'], ['#93C5FD', '#3B82F6', '#1E40AF'], 0.05),
            ('Line Drive', [f'{min_ev}-95', '95-100', '100+'], ['#86EFAC', '#22C55E', '#15803D'], 0.38),
            ('Fly Ball', [f'{min_ev}-95', '95-100', '100+'], ['#FCD34D', '#F97316', '#DC2626'], 0.71),
        ]

    for hit_type, labels, colors, x_start in legend_data:
        ax_legend.text(x_start, 0.8, hit_type, fontsize=14, fontweight='bold')
        for i, (label, color) in enumerate(zip(labels, colors)):
            y_pos = 0.5 - i * 0.25
            rect = patches.Rectangle((x_start, y_pos), 0.04, 0.15,
                                     facecolor=color, edgecolor='black')
            ax_legend.add_patch(rect)
            ax_legend.text(x_start + 0.05, y_pos + 0.075, f'{label} mph',
                           va='center', fontsize=11)

    # Summary at bottom
    ax_summary = fig.add_axes([0.1, 0.01, 0.8, 0.04])
    ax_summary.axis('off')
    if min_ev >= 100:
        summary_text = f'Total Quality BIP: {total}  |  100+ mph: {ev_high_count}'
    elif min_ev >= 95:
        summary_text = f'Total Quality BIP: {total}  |  {min_ev}-100 mph: {ev_mid_count}  |  100+ mph: {ev_high_count}'
    else:
        summary_text = f'Total Quality BIP: {total}  |  {min_ev}-95 mph: {ev_low_count}  |  95-100 mph: {ev_mid_count}  |  100+ mph: {ev_high_count}'
    ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center',
                    fontsize=13, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#f3f4f6', alpha=0.8))

    plt.tight_layout()
    return fig


def create_hard_hit_csv(df, team=None, min_ev=90):
    """Generate hard-hit balls CSV data"""
    bip_df = filter_quality_bip(df, team=team, min_ev=min_ev)

    if len(bip_df) == 0:
        return None

    # Sort by EV descending
    export_df = bip_df.sort_values('ExitSpeed', ascending=False).copy()

    # Select columns
    export_cols = ['Batter', 'TaggedHitType', 'ExitSpeed', 'Angle', 'Distance',
                   'PlayResult', 'EVDisplay', 'Date', 'Direction', 'Bearing']
    available_cols = [c for c in export_cols if c in export_df.columns]

    export_df = export_df[available_cols].copy()

    # Rename columns
    rename_map = {
        'Batter': 'Player',
        'TaggedHitType': 'Hit_Type',
        'ExitSpeed': 'Exit_Velocity',
        'Angle': 'Launch_Angle',
        'Distance': 'Distance_ft',
        'PlayResult': 'Result',
        'EVDisplay': 'EV_Range'
    }
    export_df = export_df.rename(columns={k: v for k, v in rename_map.items() if k in export_df.columns})

    return export_df


# =============================================================================
# TRAJECTORY CALCULATION - From pitch_count.py
# =============================================================================
def trajectory_9p_quadratic(pitch_data, num_points=50):
    """Calculate trajectory using the 9-parameter quadratic model."""
    x0 = pitch_data['x0']
    y0 = pitch_data['y0'] if 'y0' in pitch_data and pd.notna(pitch_data.get('y0')) else 50.0
    z0 = pitch_data['z0']
    vx0 = pitch_data['vx0']
    vy0 = pitch_data['vy0']
    vz0 = pitch_data['vz0']
    ax = pitch_data['ax0']
    ay = pitch_data['ay0']
    az = pitch_data['az0']

    if any(pd.isna(v) for v in [x0, z0, vx0, vy0, vz0, ax, ay, az]):
        return None, None, None

    a_coef = 0.5 * ay
    b_coef = vy0
    c_coef = y0 - PLATE_Y

    discriminant = b_coef ** 2 - 4 * a_coef * c_coef
    if discriminant < 0:
        return None, None, None

    t_flight = (-b_coef - np.sqrt(discriminant)) / (2 * a_coef)
    if t_flight <= 0:
        t_flight = (-b_coef + np.sqrt(discriminant)) / (2 * a_coef)

    if t_flight <= 0 or t_flight > 1.2:
        return None, None, None

    t = np.linspace(0, t_flight, num_points)

    x = x0 + vx0 * t + 0.5 * ax * t ** 2
    y = y0 + vy0 * t + 0.5 * ay * t ** 2
    z = z0 + vz0 * t + 0.5 * az * t ** 2

    return x, y, z


def trajectory_ode_physics(pitch_data, num_points=50, dt=0.001):
    """Calculate trajectory using full ODE integration with UMBA physics model."""
    x0 = pitch_data['x0']
    y0 = pitch_data['y0'] if 'y0' in pitch_data and pd.notna(pitch_data.get('y0')) else 50.0
    z0 = pitch_data['z0']
    vx0 = pitch_data['vx0']
    vy0 = pitch_data['vy0']
    vz0 = pitch_data['vz0']

    spin_rate = pitch_data.get('SpinRate', np.nan)
    spin_axis = pitch_data.get('SpinAxis', np.nan)

    if any(pd.isna(v) for v in [x0, z0, vx0, vy0, vz0]):
        return None, None, None

    if pd.isna(spin_rate) or pd.isna(spin_axis) or spin_rate < 100:
        return trajectory_9p_quadratic(pitch_data, num_points)

    omega_total = spin_rate * 0.104719754
    spin_axis_rad = np.radians(spin_axis)
    omega_x = -omega_total * np.cos(spin_axis_rad)
    omega_y = 0
    omega_z = omega_total * np.sin(spin_axis_rad)
    r = BALL_DIAMETER / 2

    def derivatives(state):
        x, y, z, vx, vy, vz = state
        v_mag = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
        if v_mag < 1e-6:
            return np.array([vx, vy, vz, 0, 0, -GRAVITY])

        omega_mag = np.sqrt(omega_x ** 2 + omega_y ** 2 + omega_z ** 2)
        if omega_mag < 1e-6:
            omega_mag = 1e-6

        S = (r * omega_mag) / v_mag
        Cl = 1.0 / (2.32 + 0.4 / S) if S > 0.01 else 0

        ax_drag = -C0_AERO * CD_CONSTANT * v_mag * vx
        ay_drag = -C0_AERO * CD_CONSTANT * v_mag * vy
        az_drag = -C0_AERO * CD_CONSTANT * v_mag * vz

        cross_x = omega_y * vz - omega_z * vy
        cross_y = omega_z * vx - omega_x * vz
        cross_z = omega_x * vy - omega_y * vx

        magnus_factor = C0_AERO * (Cl / omega_mag) * v_mag
        ax_magnus = magnus_factor * cross_x
        ay_magnus = magnus_factor * cross_y
        az_magnus = magnus_factor * cross_z

        ax = ax_drag + ax_magnus
        ay = ay_drag + ay_magnus
        az = az_drag + az_magnus - GRAVITY

        return np.array([vx, vy, vz, ax, ay, az])

    state = np.array([x0, y0, z0, vx0, vy0, vz0], dtype=float)
    trajectory = [state.copy()]
    t = 0
    max_time = 0.6

    while state[1] > PLATE_Y and t < max_time:
        k1 = derivatives(state)
        k2 = derivatives(state + 0.5 * dt * k1)
        k3 = derivatives(state + 0.5 * dt * k2)
        k4 = derivatives(state + dt * k3)
        state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory.append(state.copy())
        t += dt

    trajectory = np.array(trajectory)
    if len(trajectory) < 2:
        return None, None, None

    indices = np.linspace(0, len(trajectory) - 1, num_points).astype(int)
    x = trajectory[indices, 0]
    y = trajectory[indices, 1]
    z = trajectory[indices, 2]

    return x, y, z


def calculate_single_trajectory(pitch_data, method='9p', num_points=50):
    """Calculate trajectory for a single pitch using specified method."""
    if method == 'ode':
        return trajectory_ode_physics(pitch_data, num_points)
    else:
        return trajectory_9p_quadratic(pitch_data, num_points)


# =============================================================================
# ZONE CALCULATION HELPERS - From pitch_count.py
# =============================================================================
def mahalanobis_filter(points, threshold=2.5):
    """Filter outliers using Mahalanobis distance."""
    if len(points) < 4:
        return points, np.ones(len(points), dtype=bool)

    mean = np.mean(points, axis=0)
    cov = np.cov(points.T)

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    diff = points - mean
    mahal_dist = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
    mask = mahal_dist <= threshold

    if np.sum(mask) < 3:
        sorted_indices = np.argsort(mahal_dist)[:3]
        mask = np.zeros(len(points), dtype=bool)
        mask[sorted_indices] = True

    return points[mask], mask


# =============================================================================
# PITCHER TRAJECTORY REPORT - REPLACED VERSION (from pitch_count.py)
# =============================================================================
def get_averaged_trajectory(pitch_group, method='9p', num_points=50):
    """Calculate average trajectory for a pitch type."""
    trajectories_x = []
    trajectories_y = []
    trajectories_z = []
    plate_locs = []

    for idx, pitch in pitch_group.iterrows():
        x, y, z = calculate_single_trajectory(pitch, method=method, num_points=num_points)
        if x is not None and len(x) == num_points:
            trajectories_x.append(x)
            trajectories_y.append(y)
            trajectories_z.append(z)
            plate_locs.append({
                'height': pitch['PlateLocHeight'],
                'side': pitch['PlateLocSide']
            })

    if len(trajectories_x) == 0:
        return None, None, None

    avg_x = np.mean(trajectories_x, axis=0)
    avg_y = np.mean(trajectories_y, axis=0)
    avg_z = np.mean(trajectories_z, axis=0)

    heights = [p['height'] for p in plate_locs]
    sides = [p['side'] for p in plate_locs]

    plate_loc_data = {
        'height_mean': np.mean(heights),
        'height_std': np.std(heights),
        'side_mean': np.mean(sides),
        'side_std': np.std(sides),
        'count': len(trajectories_x)
    }

    return (avg_x, avg_y, avg_z), plate_loc_data, len(trajectories_x)


def plot_side_view_trajectory(ax, pitcher_df, pitcher_name, handedness=""):
    """Plot side view showing pitch arc from 1st base side perspective"""
    title = f'Side View - 1B ({handedness})' if handedness else 'Side View - 1B'
    ax.set_title(title, fontsize=12, fontweight='bold', color='white', pad=10)

    # Draw sky gradient
    ax.axhspan(0, 8, color='#1a3a52', alpha=0.3)

    # Draw grass
    ax.axhspan(-0.5, 0, color='#2E7D32', alpha=0.5)

    # Draw dirt area near plate and mound
    ax.fill_between([48, 65], -0.3, 0, color='#8B4513', alpha=0.4)
    ax.fill_between([-2, 5], -0.3, 0, color='#8B4513', alpha=0.4)

    # Draw mound
    mound_x = [55, 58, 60.5, 63, 66]
    mound_y = [0, 0.3, 0.5, 0.3, 0]
    ax.fill(mound_x, mound_y, color='#8B4513', alpha=0.6)

    # Draw rubber
    ax.plot([60, 61], [0.55, 0.55], 'w-', linewidth=4, solid_capstyle='butt')

    # Draw home plate
    ax.plot([-0.5, 0.5], [0, 0], 'w-', linewidth=3)

    # Draw strike zone at plate
    ax.fill_between([-0.3, 0.3], STRIKE_ZONE_HEIGHT_LOW, STRIKE_ZONE_HEIGHT_HIGH,
                    alpha=0.2, color='white')
    ax.plot([-0.3, 0.3, 0.3, -0.3, -0.3],
            [STRIKE_ZONE_HEIGHT_LOW, STRIKE_ZONE_HEIGHT_LOW,
             STRIKE_ZONE_HEIGHT_HIGH, STRIKE_ZONE_HEIGHT_HIGH, STRIKE_ZONE_HEIGHT_LOW],
            'w-', linewidth=1.5, alpha=0.7)

    release_distances = []

    for pitch_type in pitcher_df['TaggedPitchType'].unique():
        if pd.isna(pitch_type):
            continue

        pitch_group = pitcher_df[pitcher_df['TaggedPitchType'] == pitch_type]
        trajectory_data, plate_loc_data, count = get_averaged_trajectory(pitch_group)

        if trajectory_data is None:
            continue

        x, y, z = trajectory_data
        color = PITCH_COLORS.get(pitch_type, PITCH_COLORS['Other'])
        release_distances.append(y[0])

        ax.plot(y, z, color=color, linewidth=3, alpha=0.9)

        ball_indices = np.linspace(0, len(y) - 1, 6).astype(int)
        for idx in ball_indices:
            ax.scatter(y[idx], z[idx], s=100, color='white',
                       edgecolors=color, linewidth=2, alpha=0.9, zorder=5)

    if release_distances:
        avg_release = np.mean(release_distances)
        ax.text(avg_release, 7.2, f'Release\n~{avg_release:.0f} ft',
                ha='center', fontsize=9, color='white', alpha=0.7)

    ax.text(0, -0.5, 'Plate', ha='center', fontsize=9, color='white', alpha=0.7)

    ax.set_xlim(-5, 55)
    ax.set_ylim(-0.8, 7.5)
    ax.set_xlabel('Distance from Plate (ft)', fontsize=10, color='white')
    ax.set_ylabel('Height (ft)', fontsize=10, color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.grid(True, alpha=0.15, color='white')
    ax.invert_xaxis()


def plot_catcher_view_trajectory(ax, pitcher_df, pitcher_name, handedness=""):
    """Plot catcher's view with strike zone, trajectories, and KDE zones."""
    title = f"Catcher's View ({handedness})" if handedness else "Catcher's View"
    ax.set_title(title, fontsize=12, fontweight='bold', color='white', pad=10)

    # Draw background
    ax.axhspan(-1, 8, color='#3d2817', alpha=0.3)

    # Draw strike zone
    strike_zone = Rectangle((-STRIKE_ZONE_WIDTH / 2, STRIKE_ZONE_HEIGHT_LOW),
                            STRIKE_ZONE_WIDTH,
                            STRIKE_ZONE_HEIGHT_HIGH - STRIKE_ZONE_HEIGHT_LOW,
                            fill=True, facecolor='#ffffff', alpha=0.15,
                            edgecolor='white', linewidth=2)
    ax.add_patch(strike_zone)

    # Draw strike zone grid (9 sections)
    for i in range(1, 3):
        h = STRIKE_ZONE_HEIGHT_LOW + i * (STRIKE_ZONE_HEIGHT_HIGH - STRIKE_ZONE_HEIGHT_LOW) / 3
        ax.plot([-STRIKE_ZONE_WIDTH / 2, STRIKE_ZONE_WIDTH / 2], [h, h],
                'w-', alpha=0.3, linewidth=1)
        v = -STRIKE_ZONE_WIDTH / 2 + i * STRIKE_ZONE_WIDTH / 3
        ax.plot([v, v], [STRIKE_ZONE_HEIGHT_LOW, STRIKE_ZONE_HEIGHT_HIGH],
                'w-', alpha=0.3, linewidth=1)

    # Draw home plate
    plate_points = np.array([
        [-PLATE_WIDTH / 2, 1.0],
        [PLATE_WIDTH / 2, 1.0],
        [PLATE_WIDTH / 2, 0.85],
        [0, 0.7],
        [-PLATE_WIDTH / 2, 0.85],
        [-PLATE_WIDTH / 2, 1.0]
    ])
    plate = Polygon(plate_points, closed=True, facecolor='white',
                    edgecolor='#333', linewidth=2, alpha=0.9)
    ax.add_patch(plate)

    # Add 1B/3B labels
    ax.text(-2.0, 0.6, '3B', fontsize=10, color='white', alpha=0.6, ha='center')
    ax.text(2.0, 0.6, '1B', fontsize=10, color='white', alpha=0.6, ha='center')

    # Collect pitch data for each type
    pitch_data_list = []

    for pitch_type in pitcher_df['TaggedPitchType'].unique():
        if pd.isna(pitch_type):
            continue

        pitch_group = pitcher_df[pitcher_df['TaggedPitchType'] == pitch_type]
        trajectory_data, plate_loc_data, count = get_averaged_trajectory(pitch_group)

        if trajectory_data is None or plate_loc_data is None:
            continue

        x, y, z = trajectory_data
        color = PITCH_COLORS.get(pitch_type, PITCH_COLORS['Other'])

        # Get individual pitch locations for zone calculation
        sides = pitch_group['PlateLocSide'].dropna().values
        heights = pitch_group['PlateLocHeight'].dropna().values

        # Negate sides to match trajectory x-coordinate convention
        sides_negated = -sides

        pitch_data_list.append({
            'pitch_type': pitch_type,
            'x': x, 'y': y, 'z': z,
            'color': color,
            'final_x': x[-1],
            'final_z': z[-1],
            'sides': sides_negated,
            'heights': heights,
            'avg_side': -plate_loc_data['side_mean'],
            'avg_height': plate_loc_data['height_mean'],
            'count': count
        })

    # Sort by count (draw smaller samples on top for visibility)
    pitch_data_list.sort(key=lambda d: d['count'], reverse=True)

    # Draw KDE zones
    for data in pitch_data_list:
        sides = data['sides']
        heights = data['heights']
        color = data['color']

        if len(sides) < 4:
            continue

        points = np.column_stack([sides, heights])
        filtered_points, mask = mahalanobis_filter(points, threshold=2.5)

        if len(filtered_points) < 4:
            continue

        sides_filt = filtered_points[:, 0]
        heights_filt = filtered_points[:, 1]

        try:
            positions = np.vstack([sides_filt, heights_filt])
            kernel = stats.gaussian_kde(positions, bw_method='scott')
            kernel.set_bandwidth(kernel.factor * 1.2)

            x_margin = 0.3
            y_margin = 0.3
            x_grid = np.linspace(sides_filt.min() - x_margin,
                                sides_filt.max() + x_margin, 80)
            y_grid = np.linspace(heights_filt.min() - y_margin,
                                heights_filt.max() + y_margin, 80)
            X, Y = np.meshgrid(x_grid, y_grid)
            positions_grid = np.vstack([X.ravel(), Y.ravel()])
            Z = kernel(positions_grid).reshape(X.shape)

            z_flat = Z.flatten()
            z_sorted = np.sort(z_flat)[::-1]
            cumsum = np.cumsum(z_sorted)
            cumsum_norm = cumsum / cumsum[-1]

            target_percentile = 0.25
            idx = np.searchsorted(cumsum_norm, target_percentile)
            if idx < len(z_sorted):
                contour_level = z_sorted[idx]
            else:
                contour_level = z_sorted[-1]

            ax.contourf(X, Y, Z, levels=[contour_level, Z.max()],
                       colors=[color], alpha=0.35, zorder=1)
            ax.contour(X, Y, Z, levels=[contour_level],
                      colors=[color], alpha=0.7, linewidths=2.0, zorder=2)

        except Exception:
            ax.scatter(sides_filt, heights_filt, s=30, color=color, alpha=0.3, zorder=1)

    # Draw trajectories and average dots
    for data in pitch_data_list:
        x, z = data['x'], data['z']
        color = data['color']

        # Draw trajectory as fading trail
        num_trail_balls = 10
        trail_indices = np.linspace(0, len(x) - 1, num_trail_balls).astype(int)

        for i, idx in enumerate(trail_indices):
            progress = i / len(trail_indices)
            alpha = 0.1 + 0.5 * progress
            size = 30 + 120 * progress
            ax.scatter(x[idx], z[idx], s=size, color=color, alpha=alpha, zorder=3)

        # Draw trajectory line
        ax.plot(x, z, color=color, linewidth=2, alpha=0.5, zorder=2)

        # Draw average location dot
        ax.scatter(data['avg_side'], data['avg_height'], s=280, color=color,
                   edgecolors='white', linewidth=2.5, alpha=0.95, zorder=5)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(0.5, 6.25)
    ax.set_xlabel('Horizontal (ft)', fontsize=10, color='white')
    ax.set_ylabel('Height (ft)', fontsize=10, color='white')
    ax.set_aspect('equal')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.grid(True, alpha=0.15, color='white')


def create_pitcher_trajectory_report(df, pitcher_name):
    """Create pitcher trajectory report with RHH/LHH splits - NEW VERSION from pitch_count.py"""
    pitcher_df = df[df['Pitcher'] == pitcher_name].copy()

    if len(pitcher_df) == 0:
        return None

    # Required columns for trajectory
    required_cols = ['x0', 'z0', 'vx0', 'vy0', 'vz0', 'ax0', 'ay0', 'az0',
                     'PlateLocHeight', 'PlateLocSide']

    pitcher_df = pitcher_df.dropna(subset=required_cols)

    if len(pitcher_df) == 0:
        return None

    # Combine ChangeUp and Splitter
    changeup_count = len(pitcher_df[pitcher_df['TaggedPitchType'] == 'ChangeUp'])
    splitter_count = len(pitcher_df[pitcher_df['TaggedPitchType'] == 'Splitter'])
    if changeup_count > 0 or splitter_count > 0:
        if changeup_count >= splitter_count:
            pitcher_df.loc[pitcher_df['TaggedPitchType'] == 'Splitter', 'TaggedPitchType'] = 'ChangeUp'
        else:
            pitcher_df.loc[pitcher_df['TaggedPitchType'] == 'ChangeUp', 'TaggedPitchType'] = 'Splitter'

    # Split by batter handedness
    rhh_df = pitcher_df[pitcher_df['BatterSide'] == 'Right'].copy()
    lhh_df = pitcher_df[pitcher_df['BatterSide'] == 'Left'].copy()

    # Create figure with dark background
    fig = plt.figure(figsize=(16, 12), facecolor='#1a1a2e')
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], wspace=0.12, hspace=0.20)

    # TOP ROW: vs RHH
    ax1_rhh = fig.add_subplot(gs[0, 0], facecolor='#1a1a2e')
    ax2_rhh = fig.add_subplot(gs[0, 1], facecolor='#1a1a2e')

    if len(rhh_df) > 0:
        plot_side_view_trajectory(ax1_rhh, rhh_df, pitcher_name, "vs RHH")
        plot_catcher_view_trajectory(ax2_rhh, rhh_df, pitcher_name, "vs RHH")
    else:
        ax1_rhh.set_title("Side View - 1B (vs RHH)", fontsize=12, fontweight='bold', color='white')
        ax1_rhh.text(0.5, 0.5, 'No Data', transform=ax1_rhh.transAxes,
                    fontsize=20, color='white', alpha=0.5, ha='center', va='center')
        ax2_rhh.set_title("Catcher's View (vs RHH)", fontsize=12, fontweight='bold', color='white')
        ax2_rhh.text(0.5, 0.5, 'No Data', transform=ax2_rhh.transAxes,
                    fontsize=20, color='white', alpha=0.5, ha='center', va='center')

    # BOTTOM ROW: vs LHH
    ax1_lhh = fig.add_subplot(gs[1, 0], facecolor='#1a1a2e')
    ax2_lhh = fig.add_subplot(gs[1, 1], facecolor='#1a1a2e')

    if len(lhh_df) > 0:
        plot_side_view_trajectory(ax1_lhh, lhh_df, pitcher_name, "vs LHH")
        plot_catcher_view_trajectory(ax2_lhh, lhh_df, pitcher_name, "vs LHH")
    else:
        ax1_lhh.set_title("Side View - 1B (vs LHH)", fontsize=12, fontweight='bold', color='white')
        ax1_lhh.text(0.5, 0.5, 'No Data', transform=ax1_lhh.transAxes,
                    fontsize=20, color='white', alpha=0.5, ha='center', va='center')
        ax2_lhh.set_title("Catcher's View (vs LHH)", fontsize=12, fontweight='bold', color='white')
        ax2_lhh.text(0.5, 0.5, 'No Data', transform=ax2_lhh.transAxes,
                    fontsize=20, color='white', alpha=0.5, ha='center', va='center')

    # Get pitcher handedness
    pitcher_hand = pitcher_df['PitcherThrows'].iloc[0] if 'PitcherThrows' in pitcher_df.columns else "Unknown"

    plt.suptitle(f'Pitch Trajectory Analysis - {pitcher_name} ({pitcher_hand}HP)',
                 fontsize=16, fontweight='bold', y=0.98, color='white')

    # Extract dates
    if '_source_file' in pitcher_df.columns:
        source_files = pitcher_df['_source_file'].unique()
        dates = []
        for f in source_files:
            match = re.match(r'(\d{8})', f)
            if match:
                date_str = match.group(1)
                formatted_date = f"{date_str[4:6]}/{date_str[6:8]}/{date_str[:4]}"
                dates.append(formatted_date)
        if dates:
            date_label = " | ".join(sorted(set(dates)))
            fig.text(0.5, 0.94, date_label, ha='center', fontsize=11, color='#aaaaaa')
    elif 'Date' in pitcher_df.columns:
        dates = pitcher_df['Date'].unique()
        date_label = ", ".join([str(d) for d in sorted(dates)])
        fig.text(0.5, 0.94, date_label, ha='center', fontsize=11, color='#aaaaaa')

    plt.tight_layout(rect=[0, 0.10, 1, 0.94])

    # Add legend
    pitch_types = sorted(pitcher_df['TaggedPitchType'].dropna().unique())
    rhh_elements = []
    lhh_elements = []
    total_elements = []

    for pitch_type in pitch_types:
        color = PITCH_COLORS.get(pitch_type, PITCH_COLORS['Other'])
        pitch_df_type = pitcher_df[pitcher_df['TaggedPitchType'] == pitch_type]
        total_count = len(pitch_df_type)
        lhh_count = len(pitch_df_type[pitch_df_type['BatterSide'] == 'Left'])
        rhh_count = len(pitch_df_type[pitch_df_type['BatterSide'] == 'Right'])

        rhh_elements.append(plt.Line2D([0], [0], color=color, linewidth=4,
                                       label=f'{pitch_type} (R={rhh_count})'))
        lhh_elements.append(plt.Line2D([0], [0], color=color, linewidth=4,
                                       label=f'{pitch_type} (L={lhh_count})'))
        total_elements.append(plt.Line2D([0], [0], color=color, linewidth=4,
                                         label=f'{pitch_type} (n={total_count})'))

    if pitch_types:
        leg1 = fig.legend(handles=rhh_elements, loc='lower center', ncol=len(pitch_types),
                          fontsize=9, framealpha=0.8, facecolor='#2d2d44', edgecolor='white',
                          labelcolor='white', bbox_to_anchor=(0.5, 0.045))

        leg2 = fig.legend(handles=lhh_elements, loc='lower center', ncol=len(pitch_types),
                          fontsize=9, framealpha=0.8, facecolor='#2d2d44', edgecolor='white',
                          labelcolor='white', bbox_to_anchor=(0.5, 0.022))

        leg3 = fig.legend(handles=total_elements, loc='lower center', ncol=len(pitch_types),
                          fontsize=9, framealpha=0.8, facecolor='#2d2d44', edgecolor='white',
                          labelcolor='white', bbox_to_anchor=(0.5, -0.001))

        fig.add_artist(leg1)
        fig.add_artist(leg2)

    return fig


# =============================================================================
# ORIGINAL PITCHER GRAPHIC (kept as alternative view)
# =============================================================================
def draw_zone_with_regions(ax):
    """Draw strike zone with heart and shadow regions"""
    zone = patches.Rectangle((-0.71, 1.5), 1.42, 2, linewidth=2,
                             edgecolor='black', facecolor='none')
    ax.add_patch(zone)

    heart_width = 1.42 / 3
    heart_height = 2 / 3
    heart = patches.Rectangle((-heart_width / 2, 1.5 + heart_height),
                              heart_width, heart_height,
                              facecolor='red', alpha=0.1, edgecolor='red',
                              linewidth=1, linestyle='--')
    ax.add_patch(heart)

    for x in [-0.71 / 3, 0.71 / 3]:
        ax.plot([x, x], [1.5, 3.5], 'k-', alpha=0.2, linewidth=0.5)
    for y in [1.5 + 2 / 3, 1.5 + 4 / 3]:
        ax.plot([-0.71, 0.71], [y, y], 'k-', alpha=0.2, linewidth=0.5)


# =============================================================================
# PITCHER SCRIMMAGE REPORT
# =============================================================================
def create_pitcher_scrimmage_report(df, pitcher_name):
    """Create detailed pitcher scrimmage report"""
    pitches = df[df['Pitcher'] == pitcher_name].copy()

    if len(pitches) == 0:
        return None, None

    total = len(pitches)
    strikes = pitches[pitches['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging', 'FoulBall', 'InPlay'])]
    strike_pct = len(strikes) / total * 100 if total > 0 else 0

    swings = pitches[pitches['PitchCall'].isin(['StrikeSwinging', 'FoulBall', 'InPlay'])]
    whiffs = pitches[pitches['PitchCall'] == 'StrikeSwinging']
    whiff_pct = len(whiffs) / len(swings) * 100 if len(swings) > 0 else 0

    bip_count = len(pitches[pitches['PitchCall'] == 'InPlay'])

    pitch_stats = pitches.groupby('TaggedPitchType').agg({
        'PitchNo': 'count',
        'RelSpeed': ['mean', 'max'],
        'SpinRate': 'mean',
        'InducedVertBreak': 'mean',
        'HorzBreak': 'mean'
    }).round(1)

    pitch_stats.columns = ['Count', 'Avg Velo', 'Max Velo', 'Avg Spin', 'V Break', 'H Break']
    pitch_stats['Usage %'] = (pitch_stats['Count'] / total * 100).round(1)

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')

    dates = pitches['Date'].unique()
    date_str = ', '.join(sorted([str(d) for d in dates]))
    fig.suptitle(f'{pitcher_name} - Scrimmage Report\n{date_str}', fontsize=16, fontweight='bold')

    ax1 = plt.subplot(2, 3, 1)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(0, 5)
    ax1.set_aspect('equal')
    ax1.set_title('Strike Zone (Catcher View)', fontsize=11, fontweight='bold')
    draw_zone_with_regions(ax1)

    for pitch_type in pitches['TaggedPitchType'].dropna().unique():
        type_df = pitches[pitches['TaggedPitchType'] == pitch_type]
        color = get_pitch_color(pitch_type)
        x = type_df['PlateLocSide'].dropna()
        z = type_df['PlateLocHeight'].dropna()
        if len(x) > 0:
            ax1.scatter(x, z, c=color, s=50, alpha=0.7, edgecolors='black',
                       linewidth=0.5, label=f"{pitch_type}")
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.2)

    ax2 = plt.subplot(2, 3, 4)
    rhh_df = pitches[pitches['BatterSide'] == 'Right']
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, 5)
    ax2.set_aspect('equal')
    ax2.set_title(f'vs RHH ({len(rhh_df)})', fontsize=11, fontweight='bold')
    draw_zone_with_regions(ax2)
    for pitch_type in rhh_df['TaggedPitchType'].dropna().unique():
        type_df = rhh_df[rhh_df['TaggedPitchType'] == pitch_type]
        color = get_pitch_color(pitch_type)
        x = type_df['PlateLocSide'].dropna()
        z = type_df['PlateLocHeight'].dropna()
        if len(x) > 0:
            ax2.scatter(x, z, c=color, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.grid(True, alpha=0.2)

    ax3 = plt.subplot(2, 3, 5)
    lhh_df = pitches[pitches['BatterSide'] == 'Left']
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(0, 5)
    ax3.set_aspect('equal')
    ax3.set_title(f'vs LHH ({len(lhh_df)})', fontsize=11, fontweight='bold')
    draw_zone_with_regions(ax3)
    for pitch_type in lhh_df['TaggedPitchType'].dropna().unique():
        type_df = lhh_df[lhh_df['TaggedPitchType'] == pitch_type]
        color = get_pitch_color(pitch_type)
        x = type_df['PlateLocSide'].dropna()
        z = type_df['PlateLocHeight'].dropna()
        if len(x) > 0:
            ax3.scatter(x, z, c=color, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax3.grid(True, alpha=0.2)

    ax4 = plt.subplot(2, 3, 3)
    pitch_counts = pitches['TaggedPitchType'].value_counts()
    colors = [get_pitch_color(pt) for pt in pitch_counts.index]
    ax4.pie(pitch_counts.values, labels=pitch_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax4.set_title('Pitch Mix', fontsize=11, fontweight='bold')

    ax5 = plt.subplot(2, 3, 6)
    pitch_types = pitches['TaggedPitchType'].dropna().unique()
    velo_data = []
    labels = []
    colors_box = []
    for pt in pitch_types:
        velos = pitches[pitches['TaggedPitchType'] == pt]['RelSpeed'].dropna()
        if len(velos) > 0:
            velo_data.append(velos.values)
            labels.append(pt)
            colors_box.append(get_pitch_color(pt))

    if velo_data:
        bp = ax5.boxplot(velo_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax5.set_title('Velocity Distribution', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Velocity (mph)')
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax_text = plt.subplot(2, 3, 2)
    ax_text.axis('off')
    stats_text = f"""OVERALL STATS
━━━━━━━━━━━━━━━━━━━━
Total Pitches: {total}
Strike %: {strike_pct:.1f}%
Whiff %: {whiff_pct:.1f}%
Balls in Play: {bip_count}

SPLITS
━━━━━━━━━━━━━━━━━━━━
vs RHH: {len(rhh_df)} pitches
vs LHH: {len(lhh_df)} pitches
"""
    ax_text.text(0.1, 0.9, stats_text, transform=ax_text.transAxes,
                 fontsize=11, verticalalignment='top', family='monospace')

    plt.tight_layout()
    return fig, pitch_stats


# =============================================================================
# HITTER SCRIMMAGE REPORT
# =============================================================================
def create_hitter_scrimmage_report(df, hitter_name):
    """Create detailed hitter scrimmage report"""
    pitches = df[df['Batter'] == hitter_name].copy()

    if len(pitches) == 0:
        return None, None

    total = len(pitches)
    swings = pitches[pitches['PitchCall'].isin(['StrikeSwinging', 'FoulBall', 'InPlay'])]
    whiffs = pitches[pitches['PitchCall'] == 'StrikeSwinging']
    contact = pitches[pitches['PitchCall'].isin(['FoulBall', 'InPlay'])]

    swing_pct = len(swings) / total * 100 if total > 0 else 0
    whiff_pct = len(whiffs) / len(swings) * 100 if len(swings) > 0 else 0
    contact_pct = len(contact) / len(swings) * 100 if len(swings) > 0 else 0

    bip = pitches[(pitches['PitchCall'] == 'InPlay') & (pitches['ExitSpeed'].notna())]

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')

    dates = pitches['Date'].unique()
    date_str = ', '.join(sorted([str(d) for d in dates]))
    fig.suptitle(f'{hitter_name} - Hitter Report\n{date_str}', fontsize=16, fontweight='bold')

    ax1 = plt.subplot(2, 3, 1)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(0, 5)
    ax1.set_aspect('equal')
    ax1.set_title('Pitches Seen', fontsize=11, fontweight='bold')
    draw_zone_with_regions(ax1)

    call_colors = {
        'StrikeCalled': '#e74c3c',
        'BallCalled': '#3498db',
        'StrikeSwinging': '#f39c12',
        'FoulBall': '#9b59b6',
        'InPlay': '#2ecc71'
    }
    for call, color in call_colors.items():
        call_df = pitches[pitches['PitchCall'] == call]
        x = call_df['PlateLocSide'].dropna()
        z = call_df['PlateLocHeight'].dropna()
        if len(x) > 0:
            ax1.scatter(x, z, c=color, s=50, alpha=0.7, edgecolors='black',
                       linewidth=0.5, label=f"{call} ({len(call_df)})")
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.2)

    ax2 = plt.subplot(2, 3, 2)
    bip_with_loc = pitches[(pitches['PitchCall'] == 'InPlay') &
                           pitches['Direction'].notna() &
                           pitches['Distance'].notna()]

    ax2.set_xlim(-250, 250)
    ax2.set_ylim(0, 450)
    ax2.set_aspect('equal')
    ax2.set_title(f'Spray Chart ({len(bip_with_loc)} BIP)', fontsize=11, fontweight='bold')
    ax2.set_facecolor('#2d5016')
    ax2.axis('off')

    diamond = plt.Polygon([(0, 0), (-90, 90), (0, 180), (90, 90)],
                          fill=False, edgecolor='white', linewidth=2)
    ax2.add_patch(diamond)
    arc = patches.Arc((0, 0), 500, 500, theta1=45, theta2=135, color='white', linewidth=2)
    ax2.add_patch(arc)

    result_colors = {'Out': '#95a5a6', 'Single': '#3498db', 'Double': '#f39c12',
                     'Triple': '#9b59b6', 'HomeRun': '#e74c3c', 'Error': '#1abc9c'}

    for _, ball in bip_with_loc.iterrows():
        angle_rad = np.radians(ball['Direction'])
        dist = ball['Distance']
        x = dist * np.sin(angle_rad)
        y = dist * np.cos(angle_rad)
        result = ball.get('PlayResult', 'Out')
        color = result_colors.get(result, '#95a5a6')
        ax2.plot(x, y, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=0.5)

    ax3 = plt.subplot(2, 3, 4)
    ax3.axis('off')
    stats_text = f"""PLATE DISCIPLINE
━━━━━━━━━━━━━━━━━━━━
Pitches Seen: {total}
Swing %: {swing_pct:.1f}%
Whiff %: {whiff_pct:.1f}%
Contact %: {contact_pct:.1f}%

QUALITY CONTACT
━━━━━━━━━━━━━━━━━━━━
Balls in Play: {len(bip)}"""
    if len(bip) > 0:
        stats_text += f"""
Avg Exit Velo: {bip['ExitSpeed'].mean():.1f} mph
Max Exit Velo: {bip['ExitSpeed'].max():.1f} mph"""
        if 'Angle' in bip.columns and bip['Angle'].notna().sum() > 0:
            stats_text += f"""
Avg Launch Angle: {bip['Angle'].mean():.1f}°"""

    ax3.text(0.1, 0.9, stats_text, transform=ax3.transAxes,
             fontsize=11, verticalalignment='top', family='monospace')

    ax4 = plt.subplot(2, 3, 3)
    pitch_types = pitches['TaggedPitchType'].value_counts()
    if len(pitch_types) > 0:
        colors = [get_pitch_color(pt) for pt in pitch_types.index]
        ax4.pie(pitch_types.values, labels=pitch_types.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    ax4.set_title('Pitch Types Seen', fontsize=11, fontweight='bold')

    if len(bip) > 0:
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(bip['ExitSpeed'].dropna(), bins=15, color='#3498db', edgecolor='black', alpha=0.7)
        ax5.axvline(bip['ExitSpeed'].mean(), color='red', linestyle='--', linewidth=2, label='Avg')
        ax5.set_title('Exit Velocity Distribution', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Exit Velocity (mph)')
        ax5.set_ylabel('Count')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        if 'Angle' in bip.columns and bip['Angle'].notna().sum() > 0:
            ax6 = plt.subplot(2, 3, 6)
            ax6.hist(bip['Angle'].dropna(), bins=15, color='#2ecc71', edgecolor='black', alpha=0.7)
            ax6.axvline(bip['Angle'].mean(), color='red', linestyle='--', linewidth=2, label='Avg')
            ax6.set_title('Launch Angle Distribution', fontsize=11, fontweight='bold')
            ax6.set_xlabel('Launch Angle (°)')
            ax6.set_ylabel('Count')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    stats_df = pd.DataFrame({
        'Metric': ['Pitches Seen', 'Swing %', 'Whiff %', 'Contact %', 'BIP',
                   'Avg EV', 'Max EV'],
        'Value': [total, f'{swing_pct:.1f}%', f'{whiff_pct:.1f}%', f'{contact_pct:.1f}%',
                  len(bip), f"{bip['ExitSpeed'].mean():.1f}" if len(bip) > 0 else '-',
                  f"{bip['ExitSpeed'].max():.1f}" if len(bip) > 0 else '-']
    })

    return fig, stats_df


# =============================================================================
# AT-BAT SEQUENCES PDF
# =============================================================================
def create_at_bat_pdf(df, output_path):
    """Create PDF with at-bat sequences"""
    # Group by at-bats
    abs_grouped = df.groupby(['Pitcher', 'Batter', 'Inning', 'PAofInning'])

    with PdfPages(output_path) as pdf:
        for (pitcher, batter, inning, pa), ab_df in abs_grouped:
            ab_df = ab_df.sort_values('PitchNo')

            if len(ab_df) == 0:
                continue

            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(f'{pitcher} vs {batter}\nInning {inning}, PA #{pa}',
                        fontsize=14, fontweight='bold')

            ax1 = plt.subplot(1, 2, 1)
            ax1.set_xlim(-3, 3)
            ax1.set_ylim(0, 5)
            ax1.set_aspect('equal')
            ax1.set_title("Catcher's View", fontsize=11)
            draw_zone_with_regions(ax1)

            for i, (_, pitch) in enumerate(ab_df.iterrows(), 1):
                x = pitch.get('PlateLocSide', 0)
                z = pitch.get('PlateLocHeight', 2.5)
                if pd.isna(x) or pd.isna(z):
                    continue
                color = get_pitch_color(pitch.get('TaggedPitchType', 'Other'))
                ax1.scatter(x, z, c=color, s=100, edgecolors='black', linewidth=1, zorder=5)
                ax1.annotate(str(i), (x, z), ha='center', va='center',
                           fontsize=8, fontweight='bold', color='white')
            ax1.grid(True, alpha=0.2)

            ax2 = plt.subplot(1, 2, 2)
            ax2.axis('off')

            pitch_text = "PITCH SEQUENCE\n" + "=" * 40 + "\n\n"
            for i, (_, pitch) in enumerate(ab_df.iterrows(), 1):
                pt = pitch.get('TaggedPitchType', 'Unknown')
                velo = pitch.get('RelSpeed', 0)
                call = pitch.get('PitchCall', 'Unknown')
                pitch_text += f"{i}. {pt} - {velo:.0f} mph\n   Result: {call}\n\n"

            final_result = ab_df.iloc[-1].get('PlayResult', 'Unknown')
            pitch_text += f"\nFINAL RESULT: {final_result}"

            ax2.text(0.1, 0.95, pitch_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    return output_path


# =============================================================================
# FOUL BALL ZONE REPORT
# =============================================================================
def get_zone_status(row):
    """Determine if pitch is in zone, shadow zone, or outside."""
    side = row['PlateLocSide']
    height = row['PlateLocHeight']

    in_zone = (ZONE_LEFT <= side <= ZONE_RIGHT and
               ZONE_BOTTOM <= height <= ZONE_TOP)

    if in_zone:
        return 'zone'

    in_shadow = (ZONE_LEFT - SHADOW_BUFFER <= side <= ZONE_RIGHT + SHADOW_BUFFER and
                 ZONE_BOTTOM - SHADOW_BUFFER <= height <= ZONE_TOP + SHADOW_BUFFER)

    if in_shadow:
        return 'shadow'

    return 'outside'


def draw_foul_strike_zone(ax, df_plot, title_text):
    """Draw a single strike zone with foul ball data on the given axes."""
    padding = 0.6
    ax.set_xlim(ZONE_LEFT - padding, ZONE_RIGHT + padding)
    ax.set_ylim(ZONE_BOTTOM - padding, ZONE_TOP + padding)

    # Draw shadow zone
    shadow_rect = Rectangle(
        (ZONE_LEFT - SHADOW_BUFFER, ZONE_BOTTOM - SHADOW_BUFFER),
        (ZONE_RIGHT - ZONE_LEFT) + 2 * SHADOW_BUFFER,
        (ZONE_TOP - ZONE_BOTTOM) + 2 * SHADOW_BUFFER,
        fill=True, facecolor='#f0f0f0', edgecolor='#cccccc',
        linewidth=1.5, linestyle='--', zorder=1
    )
    ax.add_patch(shadow_rect)

    # Draw strike zone
    zone_rect = Rectangle(
        (ZONE_LEFT, ZONE_BOTTOM),
        ZONE_RIGHT - ZONE_LEFT,
        ZONE_TOP - ZONE_BOTTOM,
        fill=True, facecolor='white', edgecolor='black', linewidth=2.5, zorder=2
    )
    ax.add_patch(zone_rect)

    # Draw zone grid (3x3)
    zone_width = (ZONE_RIGHT - ZONE_LEFT) / 3
    zone_height = (ZONE_TOP - ZONE_BOTTOM) / 3
    for i in range(1, 3):
        ax.plot([ZONE_LEFT + i * zone_width, ZONE_LEFT + i * zone_width],
                [ZONE_BOTTOM, ZONE_TOP], color='gray', linestyle='-', alpha=0.4, linewidth=1, zorder=3)
        ax.plot([ZONE_LEFT, ZONE_RIGHT],
                [ZONE_BOTTOM + i * zone_height, ZONE_BOTTOM + i * zone_height],
                color='gray', linestyle='-', alpha=0.4, linewidth=1, zorder=3)

    # Plot pitches
    for _, row in df_plot.iterrows():
        color = get_pitch_color(row['TaggedPitchType'])
        is_in_zone = row['zone_status'] == 'zone'

        if is_in_zone:
            ax.scatter(row['PlateLocSide'], row['PlateLocHeight'],
                       c=color, s=200, marker='o', edgecolors='black',
                       linewidths=1.5, alpha=0.9, zorder=5)
        else:
            ax.scatter(row['PlateLocSide'], row['PlateLocHeight'],
                       facecolors='none', edgecolors=color, s=200, marker='o',
                       linewidths=2.5, alpha=0.8, zorder=4)

    # Labels
    ax.set_xlabel('Horizontal Location (ft)\n← Inside (RHH) | Outside (RHH) →', fontsize=10)
    ax.set_ylabel('Vertical Location (ft)', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, zorder=0)

    # Count stats for this zone
    zone_count = len(df_plot[df_plot['zone_status'] == 'zone'])
    shadow_count = len(df_plot[df_plot['zone_status'] == 'shadow'])

    subtitle = f"● Zone: {zone_count}  |  ○ Shadow: {shadow_count}"
    ax.set_title(f"{title_text}\n{subtitle}", fontsize=12, fontweight='bold', pad=10)

    # Add pitch type breakdown as text
    if len(df_plot) > 0:
        breakdown = df_plot['TaggedPitchType'].value_counts()
        breakdown_text = "Pitch Breakdown:\n" + "\n".join([f"  {pt}: {ct}" for pt, ct in breakdown.items()])
        ax.text(0.02, 0.98, breakdown_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    return zone_count, shadow_count


def create_foul_ball_zone_report(df, batter_name):
    """Create the foul ball strike zone visualization for a specific batter.
    Shows two side-by-side zones: 0-1 strikes and 2 strikes.

    Returns:
        tuple: (matplotlib figure, stats dict) or (None, None) if no data
    """
    # Filter for this batter's foul balls
    batter_df = df[df['Batter'] == batter_name].copy()
    df_fouls = batter_df[batter_df['PitchCall'].str.contains('Foul', case=False, na=False)].copy()

    if len(df_fouls) == 0:
        return None, None

    # Apply zone status
    df_fouls['zone_status'] = df_fouls.apply(get_zone_status, axis=1)
    total_fouls = len(df_fouls)

    # Split data by strike count
    df_less_than_2 = df_fouls[df_fouls['Strikes'] < 2]
    df_plot_less_than_2 = df_less_than_2[df_less_than_2['zone_status'].isin(['zone', 'shadow'])]

    df_two_strikes = df_fouls[df_fouls['Strikes'] == 2]
    df_plot_two_strikes = df_two_strikes[df_two_strikes['zone_status'].isin(['zone', 'shadow'])]

    # Get date range
    date_range = None
    if 'Date' in df_fouls.columns:
        dates = pd.to_datetime(df_fouls['Date'], errors='coerce')
        date_min = dates.min()
        date_max = dates.max()
        if pd.notna(date_min) and pd.notna(date_max):
            date_min_str = date_min.strftime('%Y-%m-%d')
            date_max_str = date_max.strftime('%Y-%m-%d')
            if date_min_str == date_max_str:
                date_range = date_min_str
            else:
                date_range = f"{date_min_str} to {date_max_str}"

    # Create figure with two subplots side by side
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 9))

    # Draw left zone (less than 2 strikes)
    zone_count_left, shadow_count_left = draw_foul_strike_zone(
        ax_left, df_plot_less_than_2, "0-1 Strikes"
    )

    # Draw right zone (2 strikes)
    zone_count_right, shadow_count_right = draw_foul_strike_zone(
        ax_right, df_plot_two_strikes, "2 Strikes"
    )

    # Create shared legend from all pitch types
    all_pitch_types = set(df_plot_less_than_2['TaggedPitchType'].unique()) | set(df_plot_two_strikes['TaggedPitchType'].unique())
    legend_patches = []
    for pt in sorted(all_pitch_types):
        color = get_pitch_color(pt)
        patch = patches.Patch(color=color, label=pt)
        legend_patches.append(patch)

    if legend_patches:
        fig.legend(handles=legend_patches, loc='lower center', ncol=len(legend_patches),
                   fontsize=10, frameon=True, fancybox=True, shadow=True,
                   bbox_to_anchor=(0.5, 0.02))

    # Main title
    title = f"{batter_name} - Foul Balls in Zone & Shadow"
    if date_range:
        title = f"{batter_name} - {date_range}\nFoul Balls in Zone & Shadow"

    total_zone = zone_count_left + zone_count_right
    total_shadow = shadow_count_left + shadow_count_right
    subtitle = f"Total: ● Zone: {total_zone}  |  ○ Shadow: {total_shadow}  |  All Fouls: {total_fouls}"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])

    stats = {
        'in_zone': total_zone,
        'shadow': total_shadow,
        'total': total_fouls,
        'zone_0_1': zone_count_left,
        'shadow_0_1': shadow_count_left,
        'zone_2': zone_count_right,
        'shadow_2': shadow_count_right
    }

    return fig, stats


# =============================================================================
# HELPER: Get common cloud sync folder paths
# =============================================================================
def get_common_sync_paths():
    """Get list of common cloud sync folder paths that might exist on this system."""
    import platform
    home = Path.home()
    paths = []

    if platform.system() == "Darwin":  # macOS
        # Google Drive
        gdrive_path = home / "Library" / "CloudStorage"
        if gdrive_path.exists():
            for folder in gdrive_path.iterdir():
                if folder.name.startswith("GoogleDrive"):
                    # Add My Drive
                    my_drive = folder / "My Drive"
                    if my_drive.exists():
                        paths.append(("Google Drive - My Drive", str(my_drive)))

                    # Add Shared drives
                    shared_drives = folder / "Shared drives"
                    if shared_drives.exists():
                        for shared in shared_drives.iterdir():
                            if shared.is_dir():
                                paths.append((f"Google Drive - {shared.name}", str(shared)))

                    # Add shortcut folders (shared folders accessed via shortcuts)
                    shortcuts_path = folder / ".shortcut-targets-by-id"
                    if shortcuts_path.exists():
                        for shortcut_id in shortcuts_path.iterdir():
                            if shortcut_id.is_dir():
                                for subfolder in shortcut_id.iterdir():
                                    if subfolder.is_dir():
                                        paths.append((f"Google Drive - {subfolder.name}", str(subfolder)))

        # Dropbox
        dropbox_path = home / "Dropbox"
        if dropbox_path.exists():
            paths.append(("Dropbox", str(dropbox_path)))

        # OneDrive
        onedrive_path = home / "Library" / "CloudStorage"
        if onedrive_path.exists():
            for folder in onedrive_path.iterdir():
                if "OneDrive" in folder.name:
                    paths.append(("OneDrive", str(folder)))

        # iCloud
        icloud_path = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
        if icloud_path.exists():
            paths.append(("iCloud Drive", str(icloud_path)))

    elif platform.system() == "Windows":
        # Google Drive
        for drive_letter in ["G", "H", "I"]:
            gdrive_path = Path(f"{drive_letter}:/My Drive")
            if gdrive_path.exists():
                paths.append(("Google Drive", str(gdrive_path)))

        gdrive_path2 = home / "Google Drive"
        if gdrive_path2.exists():
            paths.append(("Google Drive", str(gdrive_path2)))

        # Dropbox
        dropbox_path = home / "Dropbox"
        if dropbox_path.exists():
            paths.append(("Dropbox", str(dropbox_path)))

        # OneDrive
        onedrive_path = home / "OneDrive"
        if onedrive_path.exists():
            paths.append(("OneDrive", str(onedrive_path)))

    else:  # Linux
        # Google Drive (via google-drive-ocamlfuse or similar)
        gdrive_path = home / "google-drive"
        if gdrive_path.exists():
            paths.append(("Google Drive", str(gdrive_path)))

        # Dropbox
        dropbox_path = home / "Dropbox"
        if dropbox_path.exists():
            paths.append(("Dropbox", str(dropbox_path)))

    return paths


def select_folder_dialog():
    """Open a native folder picker dialog and return the selected path."""
    if not TKINTER_AVAILABLE:
        return None
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    root.attributes('-topmost', True)  # Bring dialog to front
    folder_path = filedialog.askdirectory(
        title="Select Folder Containing CSV Files"
    )
    root.destroy()
    return folder_path if folder_path else None


# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.title("⚾ Baseball Analytics Dashboard")
    st.caption("Trackman Data Analysis Tool")

    # Sidebar - Data Loading Options
    st.sidebar.header("📁 Data Source")

    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload Files", "Load from Folder"]
    )

    df = None

    # =========================================================================
    # UPLOAD FILES
    # =========================================================================
    if data_source == "Upload Files":
        uploaded_files = st.sidebar.file_uploader(
            "Upload CSV files",
            type=['csv'],
            accept_multiple_files=True
        )

        if uploaded_files:
            df = load_csv_files(uploaded_files)

    # =========================================================================
    # LOAD FROM LOCAL FOLDER (includes cloud sync folders)
    # =========================================================================
    elif data_source == "Load from Folder":

        # Initialize session state for selected folder
        if 'selected_folder_path' not in st.session_state:
            st.session_state['selected_folder_path'] = ""

        # Browse folder button (only available when running locally with tkinter)
        st.sidebar.markdown("**📂 Select Folder:**")
        if TKINTER_AVAILABLE:
            if st.sidebar.button("🔍 Browse for Folder...", key="browse_folder_btn", type="primary"):
                selected = select_folder_dialog()
                if selected:
                    st.session_state['selected_folder_path'] = selected
                    st.rerun()
        else:
            st.sidebar.info("📤 Running on cloud - use **Upload Files** or enter path below")

        # Check for common cloud sync folders
        sync_paths = get_common_sync_paths()

        if sync_paths:
            st.sidebar.markdown("**☁️ Or Quick Select Cloud Folder:**")
            selected_sync = st.sidebar.selectbox(
                "Quick select:",
                ["Custom path..."] + [f"{name}: {path}" for name, path in sync_paths],
                key="sync_folder_select"
            )

            if selected_sync != "Custom path...":
                # Extract path from selection
                folder_path = selected_sync.split(": ", 1)[1]
            elif st.session_state['selected_folder_path']:
                folder_path = st.session_state['selected_folder_path']
            else:
                folder_path = st.sidebar.text_input(
                    "Or enter path manually:",
                    placeholder="/path/to/your/csv/folder",
                    key="custom_folder_path"
                )
        else:
            # Show the browsed folder path or allow manual input
            if st.session_state['selected_folder_path']:
                folder_path = st.session_state['selected_folder_path']
                st.sidebar.text_input(
                    "Selected Folder:",
                    value=folder_path,
                    disabled=True,
                    key="display_folder_path"
                )
                if st.sidebar.button("✖️ Clear Selection", key="clear_folder"):
                    st.session_state['selected_folder_path'] = ""
                    st.rerun()
            else:
                folder_path = st.sidebar.text_input(
                    "Or enter path manually:",
                    placeholder="/path/to/your/csv/folder",
                    key="folder_path_input"
                )

            # Show helpful tips
            with st.sidebar.expander("💡 Tips: Finding your folder"):
                st.markdown("""
                **Recommended:** Use the **Browse for Folder** button above!

                **Google Drive Desktop:**
                - Mac: `~/Library/CloudStorage/GoogleDrive-.../My Drive`
                - Windows: `G:\\My Drive` or `C:\\Users\\You\\Google Drive`

                **Dropbox:**
                - Mac/Windows: `~/Dropbox` or `C:\\Users\\You\\Dropbox`

                **OneDrive:**
                - Mac: `~/Library/CloudStorage/OneDrive-...`
                - Windows: `C:\\Users\\You\\OneDrive`
                
                **External Drive:**
                - Mac: `/Volumes/YourDriveName/folder`
                - Windows: `D:\\folder` or `E:\\folder`
                """)

        if folder_path:
            # Expand ~ to home directory
            folder_path = os.path.expanduser(folder_path)

            # Check if folder exists
            if os.path.exists(folder_path):
                st.sidebar.success(f"✅ Folder found")

                # Try to count CSV files
                csv_count = len([f for f in glob.glob(os.path.join(folder_path, '*.csv'))
                                if 'playerpositioning' not in f.lower()])
                st.sidebar.caption(f"📄 {csv_count} CSV files found")
            else:
                st.sidebar.error(f"❌ Folder not found")

            # Subfolder navigation
            if os.path.exists(folder_path):
                subfolders = [f.name for f in Path(folder_path).iterdir() if f.is_dir()]
                if subfolders:
                    subfolder = st.sidebar.selectbox(
                        "📂 Subfolder (optional):",
                        ["(root folder)"] + sorted(subfolders),
                        key="subfolder_select"
                    )
                    if subfolder != "(root folder)":
                        folder_path = os.path.join(folder_path, subfolder)
                        csv_count = len([f for f in glob.glob(os.path.join(folder_path, '*.csv'))
                                        if 'playerpositioning' not in f.lower()])
                        st.sidebar.caption(f"📄 {csv_count} CSV files in subfolder")

            # Date range selector
            st.sidebar.markdown("---")
            st.sidebar.subheader("📅 Date Range")

            use_date_filter = st.sidebar.checkbox("Filter by date range", key="folder_date_filter")

            start_date = None
            end_date = None

            if use_date_filter:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30), key="folder_start")
                    start_date = datetime.combine(start_date, datetime.min.time())
                with col2:
                    end_date = st.date_input("End Date", value=datetime.now(), key="folder_end")
                    end_date = datetime.combine(end_date, datetime.max.time())

            if st.sidebar.button("📥 Load Data", key="folder_load", type="primary"):
                with st.spinner("Loading data..."):
                    df = load_csv_from_folder(folder_path, start_date, end_date)
                    if df is not None:
                        st.session_state['df'] = df
                        st.session_state['data_source'] = 'folder'
                        st.session_state['folder_path'] = folder_path

                        # Count loaded files
                        if '_source_file' in df.columns:
                            file_count = df['_source_file'].nunique()
                            st.success(f"✅ Loaded {len(df)} rows from {file_count} files")
                        else:
                            st.success(f"✅ Loaded {len(df)} rows")
                    else:
                        st.error("No CSV files found matching your criteria")

            # Show currently loaded data info
            if 'df' in st.session_state and st.session_state.get('data_source') == 'folder':
                df = st.session_state['df']
                if '_source_file' in df.columns:
                    with st.sidebar.expander(f"📄 Loaded Files ({df['_source_file'].nunique()})"):
                        for f in sorted(df['_source_file'].unique()):
                            st.caption(f"• {f}")

    if df is None:
        st.info("👈 Please select a data source in the sidebar to get started.")

        st.markdown("""
        ### Welcome to the Baseball Analytics Dashboard!
        
        This dashboard provides several analysis tools:
        
        - **🎯 Pitcher Trajectory Report** - Side view and catcher view with KDE zones, split by RHH/LHH
        - **🏏 Team Offense Overview** - Spray chart with exit velocity visualization
        - **📋 Hard-Hit Balls List** - CSV export of quality contact
        - **👤 Hitter Scrimmage Report** - Individual hitter analysis
        - **📊 Pitcher Scrimmage Report** - Individual pitcher analysis
        - **📄 At-Bat Sequences** - PDF export of pitch sequences
        - **⚾ Foul Ball Zone Report** - Per-batter foul ball analysis in zone/shadow
        
        ---
        
        ### 📁 Data Loading Options
        
        **📤 Upload Files**  
        Drag and drop CSV files directly into the uploader.
        
        **📂 Load from Folder**  
        Point to any folder on your computer, including:
        - **Google Drive** (via Google Drive Desktop app)
        - **Dropbox** (synced folder)
        - **OneDrive** (synced folder)
        - **External hard drive**
        - **Any local folder**
        
        ---
        
        ### ☁️ Using Google Drive (No API Required!)
        
        1. Install [Google Drive for Desktop](https://www.google.com/drive/download/)
        2. Sign in with your Google account
        3. Your Drive files will sync to a local folder
        4. In this dashboard, select **"Load from Folder"**
        5. The app will auto-detect your Google Drive folder!
        
        **Common Google Drive paths:**
        - **Mac**: `~/Library/CloudStorage/GoogleDrive-you@email.com/My Drive/`
        - **Windows**: `G:\\My Drive\\` or `C:\\Users\\You\\Google Drive\\`
        
        ---
        
        ### 📅 Date Filtering
        
        Name your CSV files with the date at the start (YYYYMMDD format):
        ```
        20251003_game_data.csv
        20251015_practice.csv
        ```
        
        Then use the date filter to load only files within a specific range!
        """)
        return

    # Data Summary
    summary = get_data_summary(df)

    st.sidebar.markdown("---")
    st.sidebar.header("📈 Data Summary")
    st.sidebar.caption(f"📊 Total Pitches: {summary['total_pitches']}")
    st.sidebar.caption(f"📅 Dates: {', '.join([str(d) for d in summary['dates'][:3]])}")
    st.sidebar.caption(f"⚾ Pitchers: {len(summary['pitchers'])}")
    st.sidebar.caption(f"🏏 Batters: {len(summary['batters'])}")
    st.sidebar.caption(f"📊 Balls in Play: {summary['balls_in_play']}")

    # Report Selection
    st.sidebar.header("📊 Report Type")

    report_type = st.sidebar.selectbox(
        "Select Report",
        [
            "🎯 Pitcher Trajectory Report (RHH/LHH)",
            "🏏 Team Offense Overview (Spray Chart)",
            "📋 Hard-Hit Balls List (CSV)",
            "👤 Hitter Scrimmage Report",
            "📊 Pitcher Scrimmage Report",
            "📄 At-Bat Sequences (PDF)",
            "⚾ Foul Ball Zone Report"
        ]
    )

    st.markdown("---")

    # ==========================================================================
    # PITCHER TRAJECTORY REPORT (NEW - from pitch_count.py)
    # ==========================================================================
    if report_type == "🎯 Pitcher Trajectory Report (RHH/LHH)":
        st.header("🎯 Pitcher Trajectory Report")
        st.caption("Side view trajectories and catcher's view with KDE zones")

        pitchers = summary['pitchers']

        if not pitchers:
            st.warning("No pitchers found in data.")
        else:
            # Option to exclude certain pitchers
            with st.expander("⚙️ Filter Pitchers", expanded=False):
                excluded_pitchers = st.multiselect(
                    "Exclude pitchers from list:",
                    pitchers,
                    default=[],
                    help="Select pitchers you want to hide from the dropdown"
                )

            # Filter the pitcher list
            available_pitchers = [p for p in pitchers if p not in excluded_pitchers]

            if not available_pitchers:
                st.warning("All pitchers have been excluded. Adjust the filter above.")
            else:
                selected_pitcher = st.selectbox("Select Pitcher", available_pitchers)

                with st.spinner("Generating trajectory report..."):
                    fig = create_pitcher_trajectory_report(df, selected_pitcher)

                if fig is None:
                    st.warning(f"No trajectory data found for {selected_pitcher}. Make sure the data includes trajectory columns (x0, z0, vx0, etc.)")
                else:
                    st.pyplot(fig)
                    plt.close()

                    buf = io.BytesIO()
                    fig = create_pitcher_trajectory_report(df, selected_pitcher)
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
                    buf.seek(0)
                    st.download_button("📥 Download PNG", buf,
                                     file_name=f"pitcher_trajectory_{selected_pitcher.replace(', ', '_')}.png",
                                     mime="image/png")
                    plt.close()

    # ==========================================================================
    # TEAM OFFENSE OVERVIEW (SPRAY CHART) - FIXED
    # ==========================================================================
    elif report_type == "🏏 Team Offense Overview (Spray Chart)":
        st.header("🏏 Team Offense Overview")

        col1, col2 = st.columns([1, 3])

        with col1:
            min_ev = st.slider("Min Exit Velocity", 85, 100, 90)

            bip_df = filter_quality_bip(df, team='SAN_BRO', min_ev=min_ev)

            if len(bip_df) == 0:
                st.warning("No balls in play found matching criteria for SAN_BRO.")
            else:
                st.metric("Quality BIP", len(bip_df))
                st.metric("Avg Exit Velo", f"{bip_df['ExitSpeed'].mean():.1f} mph")
                if 'Distance' in bip_df.columns and bip_df['Distance'].notna().sum() > 0:
                    st.metric("Avg Distance", f"{bip_df['Distance'].mean():.0f} ft")

                # Show breakdown by EV range - FIXED
                st.markdown("**By Exit Velocity:**")
                ev_low = len(bip_df[(bip_df['ExitSpeed'] >= min_ev) & (bip_df['ExitSpeed'] < 95)])
                ev_mid = len(bip_df[(bip_df['ExitSpeed'] >= 95) & (bip_df['ExitSpeed'] < 100)])
                ev_high = len(bip_df[bip_df['ExitSpeed'] >= 100])
                st.caption(f"{min_ev}-95 mph: {ev_low}")
                st.caption(f"95-100 mph: {ev_mid}")
                st.caption(f"100+ mph: {ev_high}")

        with col2:
            if len(bip_df) > 0:
                fig = create_team_spray_chart(bip_df, title=f"Team Offense Overview (EV ≥ {min_ev} mph)", min_ev=min_ev)
                st.pyplot(fig)
                plt.close()

                buf = io.BytesIO()
                fig = create_team_spray_chart(bip_df, title=f"Team Offense Overview (EV ≥ {min_ev} mph)", min_ev=min_ev)
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                st.download_button("📥 Download PNG", buf, file_name="team_offense_spray.png", mime="image/png")
                plt.close()

    # ==========================================================================
    # HARD-HIT BALLS LIST (CSV)
    # ==========================================================================
    elif report_type == "📋 Hard-Hit Balls List (CSV)":
        st.header("📋 Hard-Hit Balls List")

        col1, col2 = st.columns([1, 2])

        with col1:
            min_ev = st.selectbox("Exit Velocity Threshold", [90, 95, 100])

        csv_df = create_hard_hit_csv(df, team='SAN_BRO', min_ev=min_ev)

        if csv_df is None or len(csv_df) == 0:
            st.warning(f"No SAN_BRO balls found with exit velocity ≥ {min_ev} mph")
        else:
            st.success(f"Found {len(csv_df)} hard-hit balls")

            st.subheader("Summary by Player")
            summary_df = csv_df.groupby('Player').agg({
                'Exit_Velocity': ['count', 'max', 'mean']
            }).round(1)
            summary_df.columns = ['Count', 'Max EV', 'Avg EV']
            summary_df = summary_df.sort_values('Count', ascending=False)
            st.dataframe(summary_df, use_container_width=True)

            st.subheader("All Hard-Hit Balls")
            st.dataframe(csv_df, use_container_width=True)

            csv_data = csv_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv_data,
                             file_name=f"hard_hit_{min_ev}plus.csv", mime="text/csv")

    # ==========================================================================
    # HITTER SCRIMMAGE REPORT
    # ==========================================================================
    elif report_type == "👤 Hitter Scrimmage Report":
        st.header("👤 Hitter Scrimmage Report")

        batters = summary['batters']

        if not batters:
            st.warning("No batters found in data.")
        else:
            selected_hitter = st.selectbox("Select Hitter", batters)

            fig, stats_df = create_hitter_scrimmage_report(df, selected_hitter)

            if fig is None:
                st.warning(f"No data found for {selected_hitter}")
            else:
                st.pyplot(fig)
                plt.close()

                if stats_df is not None:
                    st.subheader("Stats Summary")
                    st.dataframe(stats_df, use_container_width=True)

                buf = io.BytesIO()
                fig, _ = create_hitter_scrimmage_report(df, selected_hitter)
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                st.download_button("📥 Download PNG", buf,
                                 file_name=f"hitter_{selected_hitter.replace(', ', '_')}.png",
                                 mime="image/png")
                plt.close()

    # ==========================================================================
    # PITCHER SCRIMMAGE REPORT
    # ==========================================================================
    elif report_type == "📊 Pitcher Scrimmage Report":
        st.header("📊 Pitcher Scrimmage Report")

        pitchers = summary['pitchers']

        if not pitchers:
            st.warning("No pitchers found in data.")
        else:
            selected_pitcher = st.selectbox("Select Pitcher", pitchers)

            fig, pitch_stats = create_pitcher_scrimmage_report(df, selected_pitcher)

            if fig is None:
                st.warning(f"No data found for {selected_pitcher}")
            else:
                st.pyplot(fig)
                plt.close()

                if pitch_stats is not None:
                    st.subheader("Pitch Type Breakdown")
                    st.dataframe(pitch_stats, use_container_width=True)

                buf = io.BytesIO()
                fig, _ = create_pitcher_scrimmage_report(df, selected_pitcher)
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                st.download_button("📥 Download PNG", buf,
                                 file_name=f"pitcher_scrimmage_{selected_pitcher.replace(', ', '_')}.png",
                                 mime="image/png")
                plt.close()

    # ==========================================================================
    # AT-BAT SEQUENCES (PDF)
    # ==========================================================================
    elif report_type == "📄 At-Bat Sequences (PDF)":
        st.header("📄 At-Bat Pitch Sequence Report")

        st.info("This generates a PDF with each at-bat on a separate page, showing catcher's view and pitch data.")

        if st.button("🚀 Generate PDF Report"):
            with st.spinner("Generating PDF... This may take a minute."):
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    output_path = create_at_bat_pdf(df, tmp.name)

                    if output_path:
                        with open(output_path, 'rb') as f:
                            pdf_bytes = f.read()

                        st.success("PDF generated successfully!")
                        st.download_button("📥 Download PDF", pdf_bytes,
                                         file_name="at_bat_report.pdf",
                                         mime="application/pdf")

                        os.unlink(output_path)
                    else:
                        st.error("Failed to generate PDF. Check your data format.")

    # ==========================================================================
    # FOUL BALL ZONE REPORT
    # ==========================================================================
    elif report_type == "⚾ Foul Ball Zone Report":
        st.header("⚾ Foul Ball Zone Report")
        st.caption("Visualize foul ball locations in the strike zone and shadow zone")

        batters = summary['batters']

        if not batters:
            st.warning("No batters found in data.")
        else:
            selected_batter = st.selectbox("Select Batter", batters)

            if st.button("🔄 Generate Report", type="primary"):
                with st.spinner("Generating foul ball zone report..."):
                    fig, stats = create_foul_ball_zone_report(df, selected_batter)

                if fig is None:
                    st.warning(f"No foul ball data found for {selected_batter}")
                else:
                    st.pyplot(fig)
                    plt.close()

                    # Display stats - totals
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("In-Zone Fouls", stats['in_zone'])
                    with col2:
                        st.metric("Shadow Zone Fouls", stats['shadow'])
                    with col3:
                        st.metric("Total Fouls", stats['total'])

                    # Display stats - by strike count
                    st.divider()
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.caption("0-1 Strikes")
                        st.metric("Zone", stats['zone_0_1'])
                        st.metric("Shadow", stats['shadow_0_1'])
                    with col_b:
                        st.caption("2 Strikes")
                        st.metric("Zone", stats['zone_2'])
                        st.metric("Shadow", stats['shadow_2'])

                    # Download button
                    buf = io.BytesIO()
                    fig, _ = create_foul_ball_zone_report(df, selected_batter)
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                    buf.seek(0)
                    st.download_button("📥 Download PNG", buf,
                                     file_name=f"foul_ball_zone_{selected_batter.replace(', ', '_')}.png",
                                     mime="image/png")
                    plt.close()


if __name__ == "__main__":
    main()