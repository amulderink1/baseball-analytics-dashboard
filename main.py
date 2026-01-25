"""
⚾ Baseball Analytics Dashboard
Unified interface for all baseball analytics reports

Reports included:
- Pitcher Graphic (RHH/LHH splits, catcher view)
- At-Bat Pitch Sequences (PDF)
- Team Offense Overview (spray chart)
- Hard-Hit Balls Report (CSV)
- Pitcher Scrimmage Report
- Hitter Scrimmage Report

Run with: streamlit run main.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Polygon, Wedge, Arc
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import datetime
import tempfile
import os
import io
import warnings

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
    'Splitter': '#e67e22',
    'Curveball': '#3498db',
    'Slider': '#f1c40f',
    'Cutter': '#FF8C00',
    'Sweeper': '#FF69B4',
    'Other': '#95a5a6'
}

MOUND_DISTANCE = 60.5
PLATE_Y = 1.417
PLATE_WIDTH = 17 / 12
STRIKE_ZONE_HEIGHT_LOW = 1.5
STRIKE_ZONE_HEIGHT_HIGH = 3.5
GRAVITY = 32.174


# =============================================================================
# DATA LOADING
# =============================================================================
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
        'pitchers': sorted(df[df['PitcherTeam'] != 'WES_VAL']['Pitcher'].dropna().unique().tolist()) if 'Pitcher' in df.columns else [],
        'batters': sorted(df[df['BatterTeam'] != 'WES_VAL']['Batter'].dropna().unique().tolist()) if 'Batter' in df.columns else [],
        'teams': sorted(df['BatterTeam'].dropna().unique().tolist()) if 'BatterTeam' in df.columns else [],
        'balls_in_play': len(df[df['PitchCall'] == 'InPlay']) if 'PitchCall' in df.columns else 0
    }
    return summary


def get_pitch_color(pitch_type):
    """Get standardized color for pitch type"""
    return PITCH_COLORS.get(pitch_type, PITCH_COLORS['Other'])


# =============================================================================
# HITTING REPORTS - From hitting_overview.py
# =============================================================================
def filter_quality_bip(df, team=None, min_ev=90, exclude_team='WES_VAL'):
    """Filter for quality balls in play"""
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

    bip['EVCategory'] = pd.cut(
        bip['ExitSpeed'],
        bins=[min_ev, 95, 100, float('inf')],
        labels=[f'{min_ev}-95', '95-100', '100+'],
        right=False
    )

    return bip


def get_hit_color(hit_type, ev_category):
    """Get color based on hit type and EV category"""
    colors = {
        'GroundBall': {
            '90-95': '#93C5FD', '95-100': '#3B82F6', '100+': '#1E40AF'
        },
        'LineDrive': {
            '90-95': '#86EFAC', '95-100': '#22C55E', '100+': '#15803D'
        },
        'FlyBall': {
            '90-95': '#FCD34D', '95-100': '#F97316', '100+': '#DC2626'
        }
    }
    return colors.get(hit_type, {}).get(str(ev_category), '#9CA3AF')


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


def create_team_spray_chart(bip_df, title="Team Offense Overview"):
    """Create team spray chart visualization - from hitting_overview.py"""
    fig = plt.figure(figsize=(14, 12))

    dates = bip_df['Date'].unique() if 'Date' in bip_df.columns else []
    date_str = ', '.join(sorted([str(d) for d in dates]))

    total = len(bip_df)
    gb_count = len(bip_df[bip_df['TaggedHitType'] == 'GroundBall'])
    ld_count = len(bip_df[bip_df['TaggedHitType'] == 'LineDrive'])
    fb_count = len(bip_df[bip_df['TaggedHitType'] == 'FlyBall'])

    ev90_count = len(bip_df[bip_df['EVCategory'] == '90-95']) if 'EVCategory' in bip_df.columns else 0
    ev95_count = len(bip_df[bip_df['EVCategory'] == '95-100']) if 'EVCategory' in bip_df.columns else 0
    ev100_count = len(bip_df[bip_df['EVCategory'] == '100+']) if 'EVCategory' in bip_df.columns else 0

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
        color = get_hit_color(hit['TaggedHitType'], hit.get('EVCategory', '90-95'))

        ax_spray.plot(x, y, 'o', color=color, markersize=16,
                      markeredgecolor='white', markeredgewidth=2.5, alpha=0.85)
        ax_spray.text(x, y - 0.02, f"{hit['ExitSpeed']:.1f}",
                      ha='center', va='top', fontsize=9, fontweight='bold')

    # Stats box
    stats_text = f'Total: {total} | GB: {gb_count} | LD: {ld_count} | FB: {fb_count}\n'
    stats_text += f'90-95: {ev90_count} | 95-100: {ev95_count} | 100+: {ev100_count}'
    ax_spray.text(0.98, 0.98, stats_text, transform=ax_spray.transAxes,
                  ha='right', va='top', fontsize=12,
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Legend
    ax_legend = fig.add_axes([0.1, 0.08, 0.8, 0.14])
    ax_legend.axis('off')

    legend_data = [
        ('Ground Ball', ['90-95', '95-100', '100+'], ['#93C5FD', '#3B82F6', '#1E40AF'], 0.05),
        ('Line Drive', ['90-95', '95-100', '100+'], ['#86EFAC', '#22C55E', '#15803D'], 0.38),
        ('Fly Ball', ['90-95', '95-100', '100+'], ['#FCD34D', '#F97316', '#DC2626'], 0.71),
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
    summary_text = f'Total Quality BIP: {total}  |  90-95 mph: {ev90_count}  |  95-100 mph: {ev95_count}  |  100+ mph: {ev100_count}'
    ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center',
                    fontsize=13, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='#f3f4f6', alpha=0.8))

    plt.tight_layout()
    return fig


def create_hard_hit_csv(df, team=None, min_ev=90):
    """Generate hard-hit balls CSV data - from hitting_overview.py"""
    bip_df = filter_quality_bip(df, team=team, min_ev=min_ev)

    if len(bip_df) == 0:
        return None

    # Sort by EV descending
    export_df = bip_df.sort_values('ExitSpeed', ascending=False).copy()

    # Select columns
    export_cols = ['Batter', 'TaggedHitType', 'ExitSpeed', 'Angle', 'Distance',
                   'PlayResult', 'EVCategory', 'Date', 'Direction', 'Bearing']
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
        'EVCategory': 'EV_Range'
    }
    export_df = export_df.rename(columns={k: v for k, v in rename_map.items() if k in export_df.columns})

    return export_df


# =============================================================================
# PITCHER GRAPHIC - From change.py (RHH/LHH splits, catcher view)
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

    # Grid lines
    for x in [-0.71 / 3, 0.71 / 3]:
        ax.plot([x, x], [1.5, 3.5], 'k-', alpha=0.2, linewidth=0.5)
    for y in [1.5 + 2 / 3, 1.5 + 4 / 3]:
        ax.plot([-0.71, 0.71], [y, y], 'k-', alpha=0.2, linewidth=0.5)


def create_pitcher_graphic(df, pitcher_name):
    """Create pitcher graphic with RHH/LHH splits and catcher view - from change.py"""
    pitcher_df = df[df['Pitcher'] == pitcher_name].copy()

    if len(pitcher_df) == 0:
        return None

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('white')

    # Get date range
    dates = pitcher_df['Date'].unique()
    date_str = ', '.join(sorted([str(d) for d in dates]))

    fig.suptitle(f'{pitcher_name} - Pitcher Report\n{date_str}',
                 fontsize=18, fontweight='bold')

    # All pitches - Catcher's view
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(0, 5)
    ax1.set_aspect('equal')
    ax1.set_title('All Pitches (Catcher View)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Horizontal Location (ft)')
    ax1.set_ylabel('Vertical Location (ft)')
    draw_zone_with_regions(ax1)

    for pitch_type in pitcher_df['TaggedPitchType'].dropna().unique():
        type_df = pitcher_df[pitcher_df['TaggedPitchType'] == pitch_type]
        color = get_pitch_color(pitch_type)
        x = type_df['PlateLocSide'].dropna()
        z = type_df['PlateLocHeight'].dropna()
        if len(x) > 0:
            ax1.scatter(x, z, c=color, s=60, alpha=0.7, edgecolors='black',
                       linewidth=0.5, label=f"{pitch_type} ({len(type_df)})")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.2)

    # vs RHH
    ax2 = plt.subplot(2, 3, 4)
    rhh_df = pitcher_df[pitcher_df['BatterSide'] == 'Right']
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(0, 5)
    ax2.set_aspect('equal')
    ax2.set_title(f'vs RHH ({len(rhh_df)} pitches)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Horizontal Location (ft)')
    ax2.set_ylabel('Vertical Location (ft)')
    draw_zone_with_regions(ax2)

    for pitch_type in rhh_df['TaggedPitchType'].dropna().unique():
        type_df = rhh_df[rhh_df['TaggedPitchType'] == pitch_type]
        color = get_pitch_color(pitch_type)
        x = type_df['PlateLocSide'].dropna()
        z = type_df['PlateLocHeight'].dropna()
        if len(x) > 0:
            ax2.scatter(x, z, c=color, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.grid(True, alpha=0.2)

    # vs LHH
    ax3 = plt.subplot(2, 3, 5)
    lhh_df = pitcher_df[pitcher_df['BatterSide'] == 'Left']
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(0, 5)
    ax3.set_aspect('equal')
    ax3.set_title(f'vs LHH ({len(lhh_df)} pitches)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Horizontal Location (ft)')
    ax3.set_ylabel('Vertical Location (ft)')
    draw_zone_with_regions(ax3)

    for pitch_type in lhh_df['TaggedPitchType'].dropna().unique():
        type_df = lhh_df[lhh_df['TaggedPitchType'] == pitch_type]
        color = get_pitch_color(pitch_type)
        x = type_df['PlateLocSide'].dropna()
        z = type_df['PlateLocHeight'].dropna()
        if len(x) > 0:
            ax3.scatter(x, z, c=color, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax3.grid(True, alpha=0.2)

    # Pitch Mix pie chart
    ax4 = plt.subplot(2, 3, 3)
    pitch_counts = pitcher_df['TaggedPitchType'].value_counts()
    colors = [get_pitch_color(pt) for pt in pitch_counts.index]
    ax4.pie(pitch_counts.values, labels=pitch_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax4.set_title('Pitch Mix', fontsize=12, fontweight='bold')

    # Velocity by pitch type
    ax5 = plt.subplot(2, 3, 6)
    pitch_types = pitcher_df['TaggedPitchType'].dropna().unique()
    velo_data = []
    labels = []
    colors_box = []
    for pt in pitch_types:
        velos = pitcher_df[pitcher_df['TaggedPitchType'] == pt]['RelSpeed'].dropna()
        if len(velos) > 0:
            velo_data.append(velos.values)
            labels.append(pt)
            colors_box.append(get_pitch_color(pt))

    if velo_data:
        bp = ax5.boxplot(velo_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax5.set_title('Velocity by Pitch Type', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Velocity (mph)')
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Balls in play field map
    ax6 = plt.subplot(2, 3, 2)
    bip = pitcher_df[(pitcher_df['PitchCall'] == 'InPlay') &
                     pitcher_df['Direction'].notna() &
                     pitcher_df['Distance'].notna()]

    ax6.set_xlim(-250, 250)
    ax6.set_ylim(0, 450)
    ax6.set_aspect('equal')
    ax6.set_title(f'Balls in Play ({len(bip)})', fontsize=12, fontweight='bold')
    ax6.set_facecolor('#2d5016')
    ax6.axis('off')

    # Draw field
    diamond = plt.Polygon([(0, 0), (-90, 90), (0, 180), (90, 90)],
                          fill=False, edgecolor='white', linewidth=2)
    ax6.add_patch(diamond)
    arc = patches.Arc((0, 0), 500, 500, theta1=45, theta2=135, color='white', linewidth=2)
    ax6.add_patch(arc)

    result_colors = {'Out': '#95a5a6', 'Single': '#3498db', 'Double': '#f39c12',
                     'Triple': '#9b59b6', 'HomeRun': '#e74c3c', 'Error': '#1abc9c'}

    for _, ball in bip.iterrows():
        angle_rad = np.radians(ball['Direction'])
        dist = ball['Distance']
        x = dist * np.sin(angle_rad)
        y = dist * np.cos(angle_rad)
        result = ball.get('PlayResult', 'Out')
        color = result_colors.get(result, '#95a5a6')
        ax6.plot(x, y, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=0.5)

    plt.tight_layout()
    return fig


# =============================================================================
# PITCHER SCRIMMAGE REPORT - From trackman_analytics.py
# =============================================================================
def create_pitcher_scrimmage_report(df, pitcher_name):
    """Create detailed pitcher scrimmage report - from trackman_analytics.py"""
    pitches = df[df['Pitcher'] == pitcher_name].copy()

    if len(pitches) == 0:
        return None, None

    # Calculate stats
    total = len(pitches)
    strikes = pitches[pitches['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging', 'FoulBall', 'InPlay'])]
    strike_pct = len(strikes) / total * 100 if total > 0 else 0

    swings = pitches[pitches['PitchCall'].isin(['StrikeSwinging', 'FoulBall', 'InPlay'])]
    whiffs = pitches[pitches['PitchCall'] == 'StrikeSwinging']
    whiff_pct = len(whiffs) / len(swings) * 100 if len(swings) > 0 else 0

    bip_count = len(pitches[pitches['PitchCall'] == 'InPlay'])

    # Pitch type breakdown
    pitch_stats = pitches.groupby('TaggedPitchType').agg({
        'PitchNo': 'count',
        'RelSpeed': ['mean', 'max'],
        'SpinRate': 'mean',
        'InducedVertBreak': 'mean',
        'HorzBreak': 'mean'
    }).round(1)

    pitch_stats.columns = ['Count', 'Avg Velo', 'Max Velo', 'Avg Spin', 'V Break', 'H Break']
    pitch_stats['Usage %'] = (pitch_stats['Count'] / total * 100).round(1)

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')

    dates = pitches['Date'].unique()
    date_str = ', '.join(sorted([str(d) for d in dates]))
    fig.suptitle(f'{pitcher_name} - Scrimmage Report\n{date_str}', fontsize=16, fontweight='bold')

    # Strike zone
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

    # vs RHH
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

    # vs LHH
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

    # Pitch mix
    ax4 = plt.subplot(2, 3, 3)
    pitch_counts = pitches['TaggedPitchType'].value_counts()
    colors = [get_pitch_color(pt) for pt in pitch_counts.index]
    ax4.pie(pitch_counts.values, labels=pitch_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax4.set_title('Pitch Mix', fontsize=11, fontweight='bold')

    # Velocity distribution
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

    # Stats text
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
# HITTER SCRIMMAGE REPORT - From trackman_analytics.py
# =============================================================================
def create_hitter_scrimmage_report(df, hitter_name):
    """Create detailed hitter scrimmage report - from trackman_analytics.py"""
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

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')

    dates = pitches['Date'].unique()
    date_str = ', '.join(sorted([str(d) for d in dates]))
    fig.suptitle(f'{hitter_name} - Hitter Report\n{date_str}', fontsize=16, fontweight='bold')

    # Strike zone - pitches seen
    ax1 = plt.subplot(2, 3, 1)
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(0, 5)
    ax1.set_aspect('equal')
    ax1.set_title('Pitches Seen', fontsize=11, fontweight='bold')
    draw_zone_with_regions(ax1)

    # Color by call
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

    # Spray chart
    ax2 = plt.subplot(2, 3, 2)
    bip_with_loc = pitches[(pitches['PitchCall'] == 'InPlay') &
                           pitches['Direction'].notna() &
                           pitches['Distance'].notna()]

    ax2.set_xlim(-250, 250)
    ax2.set_ylim(0, 450)
    ax2.set_aspect('equal')
    ax2.set_title(f'Batted Ball Chart ({len(bip_with_loc)})', fontsize=11, fontweight='bold')
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
        ax2.plot(x, y, 'o', color=color, markersize=10, markeredgecolor='white', markeredgewidth=0.5)

    # Exit velo distribution
    ax3 = plt.subplot(2, 3, 4)
    if len(bip) > 0:
        ax3.hist(bip['ExitSpeed'], bins=15, color='#3498db', alpha=0.7, edgecolor='black')
        ax3.axvline(bip['ExitSpeed'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f"Avg: {bip['ExitSpeed'].mean():.1f} mph")
        ax3.set_title('Exit Velocity Distribution', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Exit Velocity (mph)')
        ax3.set_ylabel('Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No batted ball data', ha='center', va='center')
        ax3.axis('off')

    # Launch angle distribution
    ax4 = plt.subplot(2, 3, 5)
    bip_la = pitches[(pitches['PitchCall'] == 'InPlay') & pitches['Angle'].notna()]
    if len(bip_la) > 0:
        ax4.hist(bip_la['Angle'], bins=15, color='#2ecc71', alpha=0.7, edgecolor='black')
        ax4.axvline(bip_la['Angle'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f"Avg: {bip_la['Angle'].mean():.1f}°")
        ax4.axvspan(8, 32, alpha=0.2, color='yellow', label='Sweet Spot (8-32°)')
        ax4.set_title('Launch Angle Distribution', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Launch Angle (degrees)')
        ax4.set_ylabel('Count')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No launch angle data', ha='center', va='center')
        ax4.axis('off')

    # EV vs LA scatter
    ax5 = plt.subplot(2, 3, 6)
    bip_scatter = pitches[(pitches['PitchCall'] == 'InPlay') &
                          pitches['ExitSpeed'].notna() &
                          pitches['Angle'].notna()]
    if len(bip_scatter) > 0:
        scatter = ax5.scatter(bip_scatter['Angle'], bip_scatter['ExitSpeed'],
                             c=bip_scatter['Distance'].fillna(0), cmap='RdYlGn',
                             s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax5.axhspan(95, 115, xmin=0.35, xmax=0.65, alpha=0.1, color='gold')
        ax5.axvspan(8, 32, alpha=0.1, color='gold')
        ax5.set_title('Exit Velo vs Launch Angle', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Launch Angle (degrees)')
        ax5.set_ylabel('Exit Velocity (mph)')
        ax5.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Distance (ft)', fontsize=9)
    else:
        ax5.text(0.5, 0.5, 'No batted ball data', ha='center', va='center')
        ax5.axis('off')

    # Stats text
    ax_text = plt.subplot(2, 3, 3)
    ax_text.axis('off')

    stats_text = f"""OVERALL STATS
━━━━━━━━━━━━━━━━━━━━
Total Pitches: {total}
Swing %: {swing_pct:.1f}%
Whiff %: {whiff_pct:.1f}%
Contact %: {contact_pct:.1f}%
"""
    if len(bip) > 0:
        hard_hit = len(bip[bip['ExitSpeed'] >= 95])
        stats_text += f"""
BATTED BALL METRICS
━━━━━━━━━━━━━━━━━━━━
Balls in Play: {len(bip)}
Avg Exit Velo: {bip['ExitSpeed'].mean():.1f} mph
Max Exit Velo: {bip['ExitSpeed'].max():.1f} mph
Hard Hit % (≥95): {hard_hit / len(bip) * 100:.1f}%
"""
        if bip['Distance'].notna().sum() > 0:
            stats_text += f"Avg Distance: {bip['Distance'].mean():.0f} ft\n"
            stats_text += f"Max Distance: {bip['Distance'].max():.0f} ft"

    ax_text.text(0.1, 0.95, stats_text, transform=ax_text.transAxes,
                 fontsize=10, verticalalignment='top', family='monospace')

    plt.tight_layout()

    # Create summary dataframe
    summary_data = {
        'Metric': ['Total Pitches', 'Swing %', 'Whiff %', 'Contact %', 'Balls in Play',
                   'Avg Exit Velo', 'Max Exit Velo', 'Hard Hit %'],
        'Value': [total, f'{swing_pct:.1f}%', f'{whiff_pct:.1f}%', f'{contact_pct:.1f}%',
                  len(bip),
                  f"{bip['ExitSpeed'].mean():.1f}" if len(bip) > 0 else 'N/A',
                  f"{bip['ExitSpeed'].max():.1f}" if len(bip) > 0 else 'N/A',
                  f"{len(bip[bip['ExitSpeed'] >= 95]) / len(bip) * 100:.1f}%" if len(bip) > 0 else 'N/A']
    }
    summary_df = pd.DataFrame(summary_data)

    return fig, summary_df


# =============================================================================
# AT-BAT REPORT (PDF) - From V2.py
# =============================================================================
def trajectory_9p_quadratic(pitch_data, num_points=50):
    """Calculate trajectory using 9-parameter quadratic model"""
    x0 = pitch_data.get('x0')
    y0 = pitch_data.get('y0', 50.0) if pd.notna(pitch_data.get('y0')) else 50.0
    z0 = pitch_data.get('z0')
    vx0 = pitch_data.get('vx0')
    vy0 = pitch_data.get('vy0')
    vz0 = pitch_data.get('vz0')
    ax = pitch_data.get('ax0')
    ay = pitch_data.get('ay0')
    az = pitch_data.get('az0')

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

    if t_flight <= 0 or t_flight > 1.0:
        return None, None, None

    t = np.linspace(0, t_flight, num_points)
    x = x0 + vx0 * t + 0.5 * ax * t ** 2
    y = y0 + vy0 * t + 0.5 * ay * t ** 2
    z = z0 + vz0 * t + 0.5 * az * t ** 2

    return x, y, z


def draw_strike_zone_pdf(ax):
    """Draw strike zone for PDF report"""
    zone_left = -PLATE_WIDTH / 2
    zone_right = PLATE_WIDTH / 2

    zone = Rectangle((zone_left, STRIKE_ZONE_HEIGHT_LOW),
                     PLATE_WIDTH,
                     STRIKE_ZONE_HEIGHT_HIGH - STRIKE_ZONE_HEIGHT_LOW,
                     fill=False, edgecolor='white', linewidth=2, alpha=0.8)
    ax.add_patch(zone)

    for i in range(1, 3):
        y = STRIKE_ZONE_HEIGHT_LOW + i * (STRIKE_ZONE_HEIGHT_HIGH - STRIKE_ZONE_HEIGHT_LOW) / 3
        ax.plot([zone_left, zone_right], [y, y], 'w-', linewidth=0.5, alpha=0.5)

    for i in range(1, 3):
        x = zone_left + i * PLATE_WIDTH / 3
        ax.plot([x, x], [STRIKE_ZONE_HEIGHT_LOW, STRIKE_ZONE_HEIGHT_HIGH],
                'w-', linewidth=0.5, alpha=0.5)


def create_at_bat_pdf(df, output_path):
    """Create PDF report with at-bat pitch sequences - from V2.py"""
    critical_cols = ['Date', 'Inning', 'Top/Bottom', 'PAofInning', 'Batter']
    valid_mask = df[critical_cols].notna().all(axis=1)
    df_valid = df[valid_mask].copy()

    if len(df_valid) == 0:
        return None

    df_valid['AtBatID'] = (df_valid['Date'].astype(str) + '_' +
                           df_valid['Inning'].astype(int).astype(str) + '_' +
                           df_valid['Top/Bottom'].astype(str) + '_' +
                           df_valid['PAofInning'].astype(int).astype(str))

    at_bats = df_valid.groupby('AtBatID')

    with PdfPages(output_path) as pdf:
        for ab_id, ab_df in at_bats:
            ab_df = ab_df.sort_values('PitchofPA')

            fig = plt.figure(figsize=(11, 8.5), facecolor='#1a1a2e')

            batter = ab_df['Batter'].iloc[0]
            inning = ab_df['Inning'].iloc[0]
            top_bottom = ab_df['Top/Bottom'].iloc[0]
            batter_side = ab_df['BatterSide'].iloc[0] if 'BatterSide' in ab_df.columns else ''
            side_abbrev = "RHH" if batter_side == "Right" else "LHH"
            pa_of_inning = ab_df['PAofInning'].iloc[0]
            date = ab_df['Date'].iloc[0]

            play_result = ab_df['PlayResult'].iloc[-1] if pd.notna(ab_df['PlayResult'].iloc[-1]) else ''

            title = f"AB{int(pa_of_inning)} - {batter} ({side_abbrev})"
            subtitle = f"Inning {int(inning)} {top_bottom} | {date} | {len(ab_df)} Pitches"
            if play_result:
                subtitle += f" | Result: {play_result}"

            fig.suptitle(title, fontsize=14, fontweight='bold', color='white', y=0.96)
            fig.text(0.5, 0.91, subtitle, ha='center', fontsize=10, color='#aaaaaa')

            # Side view
            ax1 = fig.add_axes([0.05, 0.45, 0.9, 0.4])
            ax1.set_facecolor('#1a1a2e')
            ax1.set_title('Side View (1B Side)', fontsize=11, fontweight='bold', color='white')

            # Draw mound
            ax1.fill_between([55, 66], 0, 0.5, color='#8B4513', alpha=0.4)
            ax1.plot([60, 61], [0.55, 0.55], 'w-', linewidth=4)
            ax1.fill_between([0, 2], -0.15, 0.15, color='white', alpha=0.9)

            # Strike zone at plate
            ax1.fill_between([0, 2], STRIKE_ZONE_HEIGHT_LOW, STRIKE_ZONE_HEIGHT_HIGH, alpha=0.2, color='white')
            ax1.plot([0, 2, 2, 0, 0],
                     [STRIKE_ZONE_HEIGHT_LOW, STRIKE_ZONE_HEIGHT_LOW,
                      STRIKE_ZONE_HEIGHT_HIGH, STRIKE_ZONE_HEIGHT_HIGH, STRIKE_ZONE_HEIGHT_LOW],
                     'w-', linewidth=1.5, alpha=0.7)

            # Plot trajectories
            for idx, (_, pitch) in enumerate(ab_df.iterrows(), 1):
                pitch_type = pitch.get('TaggedPitchType', 'Other')
                color = get_pitch_color(pitch_type) if pd.notna(pitch_type) else '#95a5a6'
                x, y, z = trajectory_9p_quadratic(pitch)
                if x is not None:
                    ax1.plot(y, z, color=color, linewidth=3, alpha=0.9)
                    ball_indices = np.linspace(0, len(y) - 1, 6).astype(int)
                    for bi in ball_indices:
                        ax1.scatter(y[bi], z[bi], s=80, color='white',
                                   edgecolors=color, linewidth=2, alpha=0.9, zorder=5)

            ax1.set_xlim(-5, 55)
            ax1.set_ylim(-0.5, 7.5)
            ax1.set_xlabel('Distance from Plate (ft)', color='white')
            ax1.set_ylabel('Height (ft)', color='white')
            ax1.tick_params(colors='white')
            ax1.grid(True, alpha=0.15, color='white')
            ax1.invert_xaxis()

            # Catcher's view
            ax2 = fig.add_axes([0.05, 0.08, 0.45, 0.35])
            ax2.set_facecolor('#1a1a2e')
            ax2.set_title("Catcher's View", fontsize=11, fontweight='bold', color='white')

            draw_strike_zone_pdf(ax2)

            for idx, (_, pitch) in enumerate(ab_df.iterrows(), 1):
                pitch_type = pitch.get('TaggedPitchType', 'Other')
                color = get_pitch_color(pitch_type) if pd.notna(pitch_type) else '#95a5a6'

                x = pitch.get('PlateLocSide', 0)
                z = pitch.get('PlateLocHeight', 2.5)

                if pd.notna(x) and pd.notna(z):
                    ax2.scatter(x, z, c=color, s=200, alpha=0.8,
                               edgecolors='white', linewidths=2, zorder=5)
                    ax2.text(x, z, str(idx), ha='center', va='center',
                            fontsize=10, fontweight='bold', color='white', zorder=6)

            ax2.set_xlim(-2.5, 2.5)
            ax2.set_ylim(0, 5)
            ax2.tick_params(colors='white')

            # Pitch table
            ax3 = fig.add_axes([0.55, 0.08, 0.4, 0.35])
            ax3.axis('off')

            columns = ['#', 'Type', 'Velo', 'Call', 'EV', 'LA']
            table_data = []
            for idx, (_, pitch) in enumerate(ab_df.iterrows(), 1):
                pt = str(pitch.get('TaggedPitchType', '-'))[:8] if pd.notna(pitch.get('TaggedPitchType')) else '-'
                velo = f"{pitch['RelSpeed']:.0f}" if pd.notna(pitch.get('RelSpeed')) else '-'
                call = str(pitch.get('PitchCall', '-'))[:10] if pd.notna(pitch.get('PitchCall')) else '-'
                ev = f"{pitch['ExitSpeed']:.0f}" if pd.notna(pitch.get('ExitSpeed')) else '-'
                la = f"{pitch['Angle']:.0f}" if pd.notna(pitch.get('Angle')) else '-'
                table_data.append([str(idx), pt, velo, call, ev, la])

            if table_data:
                table = ax3.table(cellText=table_data, colLabels=columns,
                                 cellLoc='center', loc='upper center',
                                 colWidths=[0.08, 0.22, 0.14, 0.26, 0.14, 0.14])
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.8)

                for j in range(len(columns)):
                    cell = table[(0, j)]
                    cell.set_facecolor('#3d3d5c')
                    cell.set_text_props(color='white', fontweight='bold')

                for i, row in enumerate(table_data):
                    pitch_type = row[1]
                    color = get_pitch_color(pitch_type)
                    for j in range(len(columns)):
                        cell = table[(i + 1, j)]
                        cell.set_facecolor('#2d2d44')
                        cell.set_text_props(color='white')
                        if j == 1:
                            cell.set_facecolor(color)

            pdf.savefig(fig, facecolor='#1a1a2e')
            plt.close(fig)

    return output_path


# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():
    st.title("⚾ Baseball Analytics Dashboard")
    st.markdown("*Unified interface for hitting and pitching analytics*")

    # Sidebar - Data Upload
    st.sidebar.header("📁 Data Source")

    uploaded_files = st.sidebar.file_uploader(
        "Upload TrackMan CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload one or more TrackMan CSV files"
    )

    if not uploaded_files:
        st.info("👆 Upload TrackMan CSV files using the sidebar to get started.")

        st.markdown("---")
        st.markdown("### Available Reports")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🏏 Hitting Reports")
            st.markdown("""
            - **Team Offense Overview** - Spray chart of hard-hit balls
            - **Hard-Hit Balls List** - CSV export of quality contact
            - **Hitter Scrimmage Report** - Individual batter analysis
            """)

        with col2:
            st.markdown("#### ⚾ Pitching Reports")
            st.markdown("""
            - **Pitcher Graphic** - RHH/LHH splits, catcher view
            - **Pitcher Scrimmage Report** - Detailed pitch analysis
            - **At-Bat Sequences (PDF)** - Each at-bat breakdown
            """)

        return

    # Load data
    df = load_csv_files(uploaded_files)

    if df is None or len(df) == 0:
        st.error("No valid data found in uploaded files.")
        return

    summary = get_data_summary(df)

    # Sidebar - Data Summary
    st.sidebar.success(f"✓ Loaded {summary['total_pitches']} pitches")
    st.sidebar.caption(f"📅 Dates: {', '.join([str(d) for d in summary['dates'][:3]])}")
    st.sidebar.caption(f"⚾ Pitchers: {len(summary['pitchers'])}")
    st.sidebar.caption(f"🏏 Batters: {len(summary['batters'])}")
    st.sidebar.caption(f"📊 Balls in Play: {summary['balls_in_play']}")

    # Sidebar - Report Selection
    st.sidebar.header("📊 Report Type")

    report_type = st.sidebar.selectbox(
        "Select Report",
        [
            "🏏 Team Offense Overview (Spray Chart)",
            "📋 Hard-Hit Balls List (CSV)",
            "👤 Hitter Scrimmage Report",
            "⚾ Pitcher Graphic (RHH/LHH)",
            "📊 Pitcher Scrimmage Report",
            "📄 At-Bat Sequences (PDF)"
        ]
    )

    st.markdown("---")

    # ==========================================================================
    # TEAM OFFENSE OVERVIEW (SPRAY CHART)
    # ==========================================================================
    if report_type == "🏏 Team Offense Overview (Spray Chart)":
        st.header("🏏 Team Offense Overview")

        col1, col2 = st.columns([1, 3])

        with col1:
            min_ev = st.slider("Min Exit Velocity", 85, 100, 90)

            bip_df = filter_quality_bip(df, min_ev=min_ev)

            if len(bip_df) == 0:
                st.warning("No balls in play found matching criteria.")
            else:
                st.metric("Quality BIP", len(bip_df))
                st.metric("Avg Exit Velo", f"{bip_df['ExitSpeed'].mean():.1f} mph")
                if 'Distance' in bip_df.columns and bip_df['Distance'].notna().sum() > 0:
                    st.metric("Avg Distance", f"{bip_df['Distance'].mean():.0f} ft")

        with col2:
            if len(bip_df) > 0:
                fig = create_team_spray_chart(bip_df, title=f"Team Offense Overview (EV ≥ {min_ev} mph)")
                st.pyplot(fig)
                plt.close()

                buf = io.BytesIO()
                fig = create_team_spray_chart(bip_df, title=f"Team Offense Overview (EV ≥ {min_ev} mph)")
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

        csv_df = create_hard_hit_csv(df, min_ev=min_ev)

        if csv_df is None or len(csv_df) == 0:
            st.warning(f"No balls found with exit velocity ≥ {min_ev} mph")
        else:
            st.success(f"Found {len(csv_df)} hard-hit balls")

            # Summary by player
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
    # PITCHER GRAPHIC (RHH/LHH)
    # ==========================================================================
    elif report_type == "⚾ Pitcher Graphic (RHH/LHH)":
        st.header("⚾ Pitcher Graphic")

        pitchers = summary['pitchers']

        if not pitchers:
            st.warning("No pitchers found in data.")
        else:
            selected_pitcher = st.selectbox("Select Pitcher", pitchers)

            fig = create_pitcher_graphic(df, selected_pitcher)

            if fig is None:
                st.warning(f"No data found for {selected_pitcher}")
            else:
                st.pyplot(fig)
                plt.close()

                buf = io.BytesIO()
                fig = create_pitcher_graphic(df, selected_pitcher)
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                st.download_button("📥 Download PNG", buf,
                                 file_name=f"pitcher_{selected_pitcher.replace(', ', '_')}.png",
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

        st.info("This generates a PDF with each at-bat on a separate page, showing side view trajectories, catcher's view, and pitch data.")

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


if __name__ == "__main__":
    main()