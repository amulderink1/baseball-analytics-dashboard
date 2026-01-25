"""
⚾ Baseball Analytics Dashboard
Unified interface for all baseball analytics reports

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Polygon, Wedge
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import ConvexHull
from scipy import stats
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

# ====================VCS menu=========================================================
# CONSTANTS
# =============================================================================
PITCH_COLORS = {
    'Sinker': '#9932CC',
    'Fastball': '#C41E3A',
    'Four-Seam': '#C41E3A',
    'Slider': '#FFD700',
    'ChangeUp': '#2E8B57',
    'Splitter': '#2E8B57',
    'Cutter': '#FF8C00',
    'Curveball': '#4169E1',
    'Sweeper': '#FF69B4',
    'Other': '#D3D3D3'
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
    return combined_df


def get_data_summary(df):
    """Get summary stats from data"""
    summary = {
        'total_pitches': len(df),
        'dates': df['Date'].unique().tolist() if 'Date' in df.columns else [],
        'pitchers': sorted(df['Pitcher'].dropna().unique().tolist()) if 'Pitcher' in df.columns else [],
        'batters': sorted(df['Batter'].dropna().unique().tolist()) if 'Batter' in df.columns else [],
        'teams': sorted(df['BatterTeam'].dropna().unique().tolist()) if 'BatterTeam' in df.columns else [],
        'balls_in_play': len(df[df['PitchCall'] == 'InPlay']) if 'PitchCall' in df.columns else 0
    }
    return summary


# =============================================================================
# HITTING REPORTS
# =============================================================================
def filter_quality_bip(df, team=None, min_ev=90):
    """Filter for quality balls in play"""
    mask = (
            (df['ExitSpeed'].notna()) &
            (df['ExitSpeed'] >= min_ev) &
            (df['TaggedHitType'].notna()) &
            (df['TaggedHitType'] != 'Undefined') &
            (df['Direction'].notna())
    )

    if team:
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


def create_spray_chart(bip_df, title="Team Hitting Overview"):
    """Create spray chart visualization"""
    fig = plt.figure(figsize=(12, 10))

    dates = bip_df['Date'].unique() if 'Date' in bip_df.columns else []
    date_str = ', '.join(sorted([str(d) for d in dates]))

    total = len(bip_df)
    gb_count = len(bip_df[bip_df['TaggedHitType'] == 'GroundBall'])
    ld_count = len(bip_df[bip_df['TaggedHitType'] == 'LineDrive'])
    fb_count = len(bip_df[bip_df['TaggedHitType'] == 'FlyBall'])

    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.96)
    if date_str:
        fig.text(0.5, 0.92, f'Date(s): {date_str}', ha='center', fontsize=12, style='italic')

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
        color = get_hit_color(hit['TaggedHitType'], hit['EVCategory'])

        ax_spray.plot(x, y, 'o', color=color, markersize=14,
                      markeredgecolor='white', markeredgewidth=2, alpha=0.85)
        ax_spray.text(x, y - 0.02, f"{hit['ExitSpeed']:.0f}",
                      ha='center', va='top', fontsize=8, fontweight='bold')

    stats_text = f'Total: {total} | GB: {gb_count} | LD: {ld_count} | FB: {fb_count}'
    ax_spray.text(0.98, 0.98, stats_text, transform=ax_spray.transAxes,
                  ha='right', va='top', fontsize=11,
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Legend
    ax_legend = fig.add_axes([0.1, 0.08, 0.8, 0.12])
    ax_legend.axis('off')

    legend_data = [
        ('Ground Ball', ['#93C5FD', '#3B82F6', '#1E40AF'], 0.05),
        ('Line Drive', ['#86EFAC', '#22C55E', '#15803D'], 0.38),
        ('Fly Ball', ['#FCD34D', '#F97316', '#DC2626'], 0.71),
    ]

    for hit_type, colors, x_start in legend_data:
        ax_legend.text(x_start, 0.8, hit_type, fontsize=12, fontweight='bold')
        for i, (label, color) in enumerate(zip(['90-95', '95-100', '100+'], colors)):
            y_pos = 0.5 - i * 0.25
            rect = patches.Rectangle((x_start, y_pos), 0.04, 0.15,
                                     facecolor=color, edgecolor='black')
            ax_legend.add_patch(rect)
            ax_legend.text(x_start + 0.05, y_pos + 0.075, f'{label} mph',
                           va='center', fontsize=10)

    plt.tight_layout()
    return fig


def create_hard_hit_report(df, team=None, min_ev=90):
    """Generate hard-hit balls report"""
    mask = (df['PitchCall'] == 'InPlay') & (df['ExitSpeed'].notna()) & (df['ExitSpeed'] >= min_ev)
    if team:
        mask = mask & (df['BatterTeam'] == team)

    report_df = df[mask].copy()

    if len(report_df) == 0:
        return None, None

    columns = ['Date', 'Batter', 'Pitcher', 'ExitSpeed', 'Angle', 'Distance',
               'PlayResult', 'TaggedHitType', 'TaggedPitchType']
    available_cols = [c for c in columns if c in report_df.columns]
    report_df = report_df[available_cols].copy()

    report_df = report_df.sort_values(['Batter', 'ExitSpeed'], ascending=[True, False])

    for col in ['ExitSpeed', 'Angle']:
        if col in report_df.columns:
            report_df[col] = report_df[col].round(1)
    if 'Distance' in report_df.columns:
        report_df['Distance'] = report_df['Distance'].round(0)

    # Summary by player
    summary = report_df.groupby('Batter').agg({
        'ExitSpeed': ['count', 'max', 'mean']
    }).round(1)
    summary.columns = ['Count', 'Max EV', 'Avg EV']
    summary = summary.sort_values('Count', ascending=False)

    return report_df, summary


def create_player_detail_report(df, player_name, team=None):
    """Generate individual player detail report"""
    mask = (df['Batter'] == player_name) & (df['PitchCall'] == 'InPlay')
    if team:
        mask = mask & (df['BatterTeam'] == team)

    player_df = df[mask].copy()

    if len(player_df) == 0:
        return None

    columns = ['Date', 'Pitcher', 'ExitSpeed', 'Angle', 'Direction', 'Distance',
               'PlayResult', 'TaggedHitType', 'TaggedPitchType', 'Inning']
    available_cols = [c for c in columns if c in player_df.columns]

    return player_df[available_cols].sort_values('ExitSpeed', ascending=False)


# =============================================================================
# PITCHING REPORTS
# =============================================================================
def trajectory_9p_quadratic(pitch_data, num_points=50):
    """Calculate trajectory using 9-parameter quadratic model"""
    x0 = pitch_data['x0']
    y0 = pitch_data.get('y0', 50.0) if pd.notna(pitch_data.get('y0')) else 50.0
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

    if t_flight <= 0 or t_flight > 1.0:
        return None, None, None

    t = np.linspace(0, t_flight, num_points)
    x = x0 + vx0 * t + 0.5 * ax * t ** 2
    y = y0 + vy0 * t + 0.5 * ay * t ** 2
    z = z0 + vz0 * t + 0.5 * az * t ** 2

    return x, y, z


def draw_strike_zone(ax, view='catcher'):
    """Draw strike zone for catcher's view"""
    zone_left = -PLATE_WIDTH / 2
    zone_right = PLATE_WIDTH / 2

    zone = Rectangle((zone_left, STRIKE_ZONE_HEIGHT_LOW),
                     PLATE_WIDTH,
                     STRIKE_ZONE_HEIGHT_HIGH - STRIKE_ZONE_HEIGHT_LOW,
                     fill=False, edgecolor='white', linewidth=2, alpha=0.8)
    ax.add_patch(zone)

    # Grid lines
    for i in range(1, 3):
        y = STRIKE_ZONE_HEIGHT_LOW + i * (STRIKE_ZONE_HEIGHT_HIGH - STRIKE_ZONE_HEIGHT_LOW) / 3
        ax.plot([zone_left, zone_right], [y, y], 'w-', linewidth=0.5, alpha=0.5)

    for i in range(1, 3):
        x = zone_left + i * PLATE_WIDTH / 3
        ax.plot([x, x], [STRIKE_ZONE_HEIGHT_LOW, STRIKE_ZONE_HEIGHT_HIGH],
                'w-', linewidth=0.5, alpha=0.5)


def create_pitcher_report(df, pitcher_name):
    """Create pitcher trajectory visualization"""
    pitcher_df = df[df['Pitcher'] == pitcher_name].copy()

    if len(pitcher_df) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#1a1a2e')

    # Catcher's view (left)
    ax1 = axes[0]
    ax1.set_facecolor('#1a1a2e')
    ax1.set_title(f"Catcher's View", fontsize=12, fontweight='bold', color='white')

    draw_strike_zone(ax1)

    # Plot pitch locations
    for pitch_type in pitcher_df['TaggedPitchType'].dropna().unique():
        type_df = pitcher_df[pitcher_df['TaggedPitchType'] == pitch_type]
        color = PITCH_COLORS.get(pitch_type, PITCH_COLORS['Other'])

        x = type_df['PlateLocSide'].dropna()
        z = type_df['PlateLocHeight'].dropna()

        if len(x) > 0 and len(z) > 0:
            ax1.scatter(x, z, c=color, s=100, alpha=0.7, label=pitch_type,
                        edgecolors='white', linewidths=1)

    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(0, 5)
    ax1.set_xlabel('Horizontal Location (ft)', color='white')
    ax1.set_ylabel('Height (ft)', color='white')
    ax1.tick_params(colors='white')
    ax1.legend(loc='upper right', fontsize=8)

    # Side view (right)
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a2e')
    ax2.set_title('Side View (1B Side)', fontsize=12, fontweight='bold', color='white')

    # Draw strike zone at plate
    ax2.fill_between([0, 2], STRIKE_ZONE_HEIGHT_LOW, STRIKE_ZONE_HEIGHT_HIGH,
                     alpha=0.2, color='white')
    ax2.plot([0, 2, 2, 0, 0],
             [STRIKE_ZONE_HEIGHT_LOW, STRIKE_ZONE_HEIGHT_LOW,
              STRIKE_ZONE_HEIGHT_HIGH, STRIKE_ZONE_HEIGHT_HIGH, STRIKE_ZONE_HEIGHT_LOW],
             'w-', linewidth=1.5, alpha=0.7)

    # Plot trajectories
    for _, pitch in pitcher_df.iterrows():
        pitch_type = pitch.get('TaggedPitchType', 'Other')
        color = PITCH_COLORS.get(pitch_type, PITCH_COLORS['Other'])

        x, y, z = trajectory_9p_quadratic(pitch)
        if x is not None:
            ax2.plot(y, z, color=color, linewidth=2, alpha=0.6)

    ax2.set_xlim(-5, 55)
    ax2.set_ylim(-0.5, 8)
    ax2.set_xlabel('Distance from Plate (ft)', color='white')
    ax2.set_ylabel('Height (ft)', color='white')
    ax2.tick_params(colors='white')
    ax2.invert_xaxis()

    # Main title
    fig.suptitle(f'Pitch Report: {pitcher_name}', fontsize=16, fontweight='bold',
                 color='white', y=0.98)

    # Stats
    total_pitches = len(pitcher_df)
    pitch_types = pitcher_df['TaggedPitchType'].value_counts()
    stats_text = f"Total: {total_pitches} pitches\n"
    for pt, count in pitch_types.items():
        if pd.notna(pt):
            stats_text += f"{pt}: {count} ({100 * count / total_pitches:.0f}%)\n"

    fig.text(0.02, 0.02, stats_text, fontsize=9, color='white',
             verticalalignment='bottom', family='monospace')

    plt.tight_layout()
    return fig


def get_pitcher_summary(df, pitcher_name):
    """Get summary statistics for a pitcher"""
    pitcher_df = df[df['Pitcher'] == pitcher_name]

    summary = {
        'Total Pitches': len(pitcher_df),
        'Pitch Types': pitcher_df['TaggedPitchType'].value_counts().to_dict(),
    }

    if 'RelSpeed' in pitcher_df.columns:
        summary['Avg Velocity'] = pitcher_df['RelSpeed'].mean()
        summary['Max Velocity'] = pitcher_df['RelSpeed'].max()

    if 'SpinRate' in pitcher_df.columns:
        summary['Avg Spin Rate'] = pitcher_df['SpinRate'].mean()

    return summary


# =============================================================================
# AT-BAT REPORT (PDF)
# =============================================================================
def create_at_bat_pdf(df, output_path):
    """Create PDF report with at-bat pitch sequences"""
    # Create unique at-bat identifier
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

            # Title info
            batter = ab_df['Batter'].iloc[0]
            inning = ab_df['Inning'].iloc[0]
            top_bottom = ab_df['Top/Bottom'].iloc[0]
            batter_side = ab_df['BatterSide'].iloc[0] if 'BatterSide' in ab_df.columns else ''
            side_abbrev = "RHH" if batter_side == "Right" else "LHH"

            play_result = ab_df['PlayResult'].iloc[-1] if pd.notna(ab_df['PlayResult'].iloc[-1]) else ''

            title = f"{batter} ({side_abbrev})"
            subtitle = f"Inning {inning} {top_bottom} | {len(ab_df)} Pitches"
            if play_result:
                subtitle += f" | Result: {play_result}"

            fig.suptitle(title, fontsize=14, fontweight='bold', color='white', y=0.96)
            fig.text(0.5, 0.91, subtitle, ha='center', fontsize=10, color='#aaaaaa')

            # Catcher's view
            ax = fig.add_axes([0.1, 0.15, 0.8, 0.7])
            ax.set_facecolor('#1a1a2e')

            draw_strike_zone(ax)

            for idx, (_, pitch) in enumerate(ab_df.iterrows(), 1):
                pitch_type = pitch.get('TaggedPitchType', 'Other')
                color = PITCH_COLORS.get(pitch_type, PITCH_COLORS['Other'])

                x = pitch.get('PlateLocSide', 0)
                z = pitch.get('PlateLocHeight', 2.5)

                if pd.notna(x) and pd.notna(z):
                    ax.scatter(x, z, c=color, s=200, alpha=0.8,
                               edgecolors='white', linewidths=2, zorder=5)
                    ax.text(x, z, str(idx), ha='center', va='center',
                            fontsize=10, fontweight='bold', color='white', zorder=6)

            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(0, 5)
            ax.set_xlabel('Horizontal Location (ft)', color='white')
            ax.set_ylabel('Height (ft)', color='white')
            ax.tick_params(colors='white')

            # Pitch table
            table_text = "# | Type | Velo | Call\n"
            table_text += "-" * 30 + "\n"
            for idx, (_, pitch) in enumerate(ab_df.iterrows(), 1):
                pt = pitch.get('TaggedPitchType', '-')[:8] if pd.notna(pitch.get('TaggedPitchType')) else '-'
                velo = f"{pitch['RelSpeed']:.0f}" if pd.notna(pitch.get('RelSpeed')) else '-'
                call = pitch.get('PitchCall', '-')[:12] if pd.notna(pitch.get('PitchCall')) else '-'
                table_text += f"{idx} | {pt:8} | {velo:4} | {call}\n"

            fig.text(0.02, 0.02, table_text, fontsize=8, color='white',
                     verticalalignment='bottom', family='monospace')

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
            - **Spray Chart** - Visual map of hard-hit balls
            - **Hard-Hit Report** - All balls ≥90 mph exit velo
            - **Launch Angle Breakdown** - By angle ranges
            - **Player Detail** - Individual batter analysis
            """)

        with col2:
            st.markdown("#### ⚾ Pitching Reports")
            st.markdown("""
            - **Pitcher Trajectory** - Side & catcher's view
            - **At-Bat Sequences** - PDF report by at-bat
            - **Pitch Mix Analysis** - Type distribution
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
            "Hitting: Spray Chart",
            "Hitting: Hard-Hit Balls",
            "Hitting: Player Detail",
            "Pitching: Trajectory Report",
            "Pitching: At-Bat Sequences (PDF)"
        ]
    )

    # Team filter
    if summary['teams']:
        selected_team = st.sidebar.selectbox(
            "Filter by Team",
            ["All Teams"] + summary['teams']
        )
        team_filter = None if selected_team == "All Teams" else selected_team
    else:
        team_filter = None

    st.markdown("---")

    # ==========================================================================
    # HITTING: SPRAY CHART
    # ==========================================================================
    if report_type == "Hitting: Spray Chart":
        st.header("🏏 Hitting Spray Chart")

        col1, col2 = st.columns([1, 3])

        with col1:
            min_ev = st.slider("Min Exit Velocity", 85, 100, 90)

            bip_df = filter_quality_bip(df, team=team_filter, min_ev=min_ev)

            if len(bip_df) == 0:
                st.warning("No balls in play found matching criteria.")
            else:
                st.metric("Quality BIP", len(bip_df))
                st.metric("Avg Exit Velo", f"{bip_df['ExitSpeed'].mean():.1f} mph")

                if 'Distance' in bip_df.columns:
                    st.metric("Avg Distance", f"{bip_df['Distance'].mean():.0f} ft")

        with col2:
            if len(bip_df) > 0:
                fig = create_spray_chart(bip_df, title=f"Hard-Hit Balls (EV ≥ {min_ev} mph)")
                st.pyplot(fig)
                plt.close()

                # Download buttons
                buf = io.BytesIO()
                fig = create_spray_chart(bip_df, title=f"Hard-Hit Balls (EV ≥ {min_ev} mph)")
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    "📥 Download PNG",
                    buf,
                    file_name="spray_chart.png",
                    mime="image/png"
                )
                plt.close()

    # ==========================================================================
    # HITTING: HARD-HIT BALLS
    # ==========================================================================
    elif report_type == "Hitting: Hard-Hit Balls":
        st.header("🔥 Hard-Hit Balls Report")

        col1, col2 = st.columns([1, 2])

        with col1:
            min_ev = st.selectbox("Exit Velocity Threshold", [90, 95, 100])

        report_df, summary_df = create_hard_hit_report(df, team=team_filter, min_ev=min_ev)

        if report_df is None:
            st.warning(f"No balls found with exit velocity ≥ {min_ev} mph")
        else:
            st.subheader("Summary by Player")
            st.dataframe(summary_df, use_container_width=True)

            st.subheader("All Hard-Hit Balls")
            st.dataframe(report_df, use_container_width=True)

            # Download CSV
            csv = report_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                file_name=f"hard_hit_{min_ev}plus.csv",
                mime="text/csv"
            )

    # ==========================================================================
    # HITTING: PLAYER DETAIL
    # ==========================================================================
    elif report_type == "Hitting: Player Detail":
        st.header("👤 Player Detail Report")

        # Filter batters by team if needed
        if team_filter:
            batters = sorted(df[df['BatterTeam'] == team_filter]['Batter'].dropna().unique())
        else:
            batters = summary['batters']

        if not batters:
            st.warning("No batters found in data.")
        else:
            selected_player = st.selectbox("Select Player", batters)

            player_df = create_player_detail_report(df, selected_player, team=team_filter)

            if player_df is None or len(player_df) == 0:
                st.warning(f"No balls in play found for {selected_player}")
            else:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Balls in Play", len(player_df))
                with col2:
                    if 'ExitSpeed' in player_df.columns:
                        st.metric("Max EV", f"{player_df['ExitSpeed'].max():.1f} mph")
                with col3:
                    if 'ExitSpeed' in player_df.columns:
                        st.metric("Avg EV", f"{player_df['ExitSpeed'].mean():.1f} mph")
                with col4:
                    if 'Distance' in player_df.columns:
                        st.metric("Max Distance", f"{player_df['Distance'].max():.0f} ft")

                st.dataframe(player_df, use_container_width=True)

                # Download
                csv = player_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    file_name=f"player_{selected_player.replace(', ', '_')}.csv",
                    mime="text/csv"
                )

    # ==========================================================================
    # PITCHING: TRAJECTORY REPORT
    # ==========================================================================
    elif report_type == "Pitching: Trajectory Report":
        st.header("⚾ Pitcher Trajectory Report")

        pitchers = summary['pitchers']

        if not pitchers:
            st.warning("No pitchers found in data.")
        else:
            selected_pitcher = st.selectbox("Select Pitcher", pitchers)

            # Pitcher summary
            pitcher_stats = get_pitcher_summary(df, selected_pitcher)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Pitches", pitcher_stats['Total Pitches'])
            with col2:
                if 'Avg Velocity' in pitcher_stats:
                    st.metric("Avg Velocity", f"{pitcher_stats['Avg Velocity']:.1f} mph")
            with col3:
                if 'Avg Spin Rate' in pitcher_stats:
                    st.metric("Avg Spin Rate", f"{pitcher_stats['Avg Spin Rate']:.0f} rpm")

            # Pitch type breakdown
            if pitcher_stats['Pitch Types']:
                st.subheader("Pitch Mix")
                pitch_type_df = pd.DataFrame.from_dict(
                    pitcher_stats['Pitch Types'],
                    orient='index',
                    columns=['Count']
                )
                pitch_type_df['Percentage'] = (pitch_type_df['Count'] / pitch_type_df['Count'].sum() * 100).round(1)
                st.dataframe(pitch_type_df, use_container_width=True)

            # Visualization
            fig = create_pitcher_report(df, selected_pitcher)
            if fig:
                st.pyplot(fig)
                plt.close()

                # Download
                buf = io.BytesIO()
                fig = create_pitcher_report(df, selected_pitcher)
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
                buf.seek(0)
                st.download_button(
                    "📥 Download PNG",
                    buf,
                    file_name=f"pitcher_{selected_pitcher.replace(', ', '_')}.png",
                    mime="image/png"
                )
                plt.close()

    # ==========================================================================
    # PITCHING: AT-BAT SEQUENCES
    # ==========================================================================
    elif report_type == "Pitching: At-Bat Sequences (PDF)":
        st.header("📄 At-Bat Pitch Sequence Report")

        st.info(
            "This generates a PDF with each at-bat shown on a separate page, including pitch locations and sequence.")

        if st.button("🚀 Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    output_path = create_at_bat_pdf(df, tmp.name)

                    if output_path:
                        with open(output_path, 'rb') as f:
                            pdf_bytes = f.read()

                        st.success("PDF generated successfully!")
                        st.download_button(
                            "📥 Download PDF",
                            pdf_bytes,
                            file_name="at_bat_report.pdf",
                            mime="application/pdf"
                        )

                        os.unlink(output_path)
                    else:
                        st.error("Failed to generate PDF. Check your data format.")


if __name__ == "__main__":
    main()
