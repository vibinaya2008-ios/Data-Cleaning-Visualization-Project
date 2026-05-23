"""
╔══════════════════════════════════════════════════════════════════╗
║         DATA CLEANING & VISUALIZATION PROJECT                    ║
║         HR Analytics · Pandas · Matplotlib · Seaborn             ║
╚══════════════════════════════════════════════════════════════════╝

PIPELINE:
  1. Generate a raw, messy dataset (missing values, outliers, duplicates)
  2. Clean it step-by-step with Pandas
  3. Visualise insights in a 9-panel dark-themed dashboard
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless rendering; swap to 'TkAgg' for GUI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════
# STEP 1 — GENERATE RAW (MESSY) DATASET
# ══════════════════════════════════════════════════════════════════
np.random.seed(42)
N = 500
DEPARTMENTS = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations']
CITIES      = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune']

df_raw = pd.DataFrame({
    'employee_id':       range(1001, 1001 + N),
    'name':              [f'Employee_{i}' for i in range(N)],
    'age':               np.random.randint(22, 62, N).astype(float),
    'department':        np.random.choice(DEPARTMENTS, N),
    'city':              np.random.choice(CITIES, N),
    'salary':            np.random.normal(75_000, 20_000, N).round(2),
    'experience_years':  np.random.randint(0, 35, N),
    'performance_score': np.random.normal(75, 15, N).round(1),
    'satisfaction_score':np.random.uniform(1, 10, N).round(1),
    'projects_completed':np.random.poisson(8, N),
})

# ── Inject missing values ──────────────────────────────────────────
for col, k in [('salary',40),('performance_score',30),
               ('age',20),('satisfaction_score',15),('department',10)]:
    df_raw.loc[np.random.choice(N, k, replace=False), col] = np.nan

# ── Inject outliers ────────────────────────────────────────────────
df_raw.loc[[5, 50, 200], 'salary']           = [350_000, -5_000, 500_000]
df_raw.loc[[10, 100],    'performance_score'] = [150, -20]

# ── Inject duplicates ──────────────────────────────────────────────
df_raw = pd.concat([df_raw, df_raw.sample(20, random_state=1)], ignore_index=True)

print("=" * 60)
print("RAW DATASET")
print(f"  Shape           : {df_raw.shape}")
print(f"  Duplicates      : {df_raw.duplicated().sum()}")
print(f"  Missing values  :\n{df_raw.isnull().sum().to_string()}")
print(f"  Salary range    : {df_raw['salary'].min():.0f} → {df_raw['salary'].max():.0f}")


# ══════════════════════════════════════════════════════════════════
# STEP 2 — DATA CLEANING
# ══════════════════════════════════════════════════════════════════
df = df_raw.copy()

# 2a. Remove exact duplicates
before = len(df)
df = df.drop_duplicates()
print(f"\nDuplicates removed : {before - len(df)}")

# 2b. Clip outliers with IQR fencing
def clip_iqr(series: pd.Series) -> pd.Series:
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return series.clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

df['salary']            = clip_iqr(df['salary'])
df['performance_score'] = df['performance_score'].clip(0, 100)

# 2c. Impute missing values
df['salary']            = df.groupby('department')['salary'] \
                            .transform(lambda x: x.fillna(x.median()))
df['performance_score'] = df['performance_score'].fillna(df['performance_score'].median())
df['age']               = df['age'].fillna(df['age'].median())
df['satisfaction_score']= df['satisfaction_score'].fillna(df['satisfaction_score'].mean())
df['department']        = df['department'].fillna(df['department'].mode()[0])

# 2d. Feature engineering
df['salary_band'] = pd.cut(df['salary'], bins=4,
                            labels=['Low', 'Mid', 'High', 'Premium'])
df['seniority']   = pd.cut(df['experience_years'], bins=[0, 2, 7, 15, 35],
                            labels=['Junior', 'Mid', 'Senior', 'Lead'])

print(f"\nCLEAN DATASET")
print(f"  Shape           : {df.shape}")
print(f"  Missing values  : {df.isnull().sum().sum()}")
print(f"  Salary range    : {df['salary'].min():.0f} → {df['salary'].max():.0f}")


# ══════════════════════════════════════════════════════════════════
# STEP 3 — VISUALISATION DASHBOARD
# ══════════════════════════════════════════════════════════════════
PALETTE = {
    'Engineering': '#6C63FF', 'Marketing': '#FF6584', 'Sales':   '#43B89C',
    'HR':          '#F7A072', 'Finance':   '#5BC0EB', 'Operations':'#FDE74C',
}
BG, CARD, TEXT, ACC = '#0F1117', '#1A1D2E', '#E8EAF6', '#6C63FF'

fig = plt.figure(figsize=(22, 26), facecolor=BG)
gs  = GridSpec(4, 3, figure=fig, hspace=0.42, wspace=0.32,
               left=0.06, right=0.97, top=0.94, bottom=0.04)

fig.text(0.5, 0.965, 'HR Analytics Dashboard', ha='center',
         fontsize=28, fontweight='bold', color=TEXT)
fig.text(0.5, 0.952, '500 employees · 6 departments · cleaned & analysed',
         ha='center', fontsize=12, color='#9E9EC8')

def style_ax(ax, title=''):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor('#2A2D3E')
    ax.xaxis.label.set_color('#9E9EC8')
    ax.yaxis.label.set_color('#9E9EC8')
    if title:
        ax.set_title(title, pad=10, color=TEXT, fontsize=12, fontweight='bold')

# ── Panel 1: Salary box by department ─────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
style_ax(ax1, '💰 Salary Distribution by Department')
dept_order = df.groupby('department')['salary'].median().sort_values(ascending=False).index
bp = ax1.boxplot(
    [df[df['department'] == d]['salary'].values for d in dept_order],
    patch_artist=True,
    medianprops=dict(color='#FDE74C', linewidth=2.5),
    whiskerprops=dict(color='#9E9EC8', linewidth=1.2),
    capprops=dict(color='#9E9EC8'),
    flierprops=dict(marker='o', markersize=3, alpha=0.4, color='#9E9EC8'),
)
for patch, dept in zip(bp['boxes'], dept_order):
    patch.set_facecolor(PALETTE[dept]); patch.set_alpha(0.75)
ax1.set_xticklabels(dept_order, rotation=15, ha='right')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x/1000:.0f}K'))
ax1.set_ylabel('Salary'); ax1.grid(axis='y', color='#2A2D3E', linewidth=0.7)

# ── Panel 2: Headcount donut ───────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(CARD)
for sp in ax2.spines.values(): sp.set_visible(False)
counts = df['department'].value_counts()
wedges, texts, autotexts = ax2.pie(
    counts, labels=counts.index, autopct='%1.1f%%',
    colors=[PALETTE[d] for d in counts.index],
    startangle=90, pctdistance=0.78,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
)
for t in texts:      t.set_color(TEXT);  t.set_fontsize(8)
for t in autotexts:  t.set_color(BG);    t.set_fontsize(8); t.set_fontweight('bold')
ax2.set_title('👥 Team Headcount', pad=10, color=TEXT, fontsize=12, fontweight='bold')

# ── Panel 3: Scatter — performance vs salary ───────────────────────
ax3 = fig.add_subplot(gs[1, :2])
style_ax(ax3, '📈 Performance Score vs Salary')
for dept in DEPARTMENTS:
    sub = df[df['department'] == dept]
    ax3.scatter(sub['performance_score'], sub['salary'],
                color=PALETTE[dept], alpha=0.55, s=28, label=dept, edgecolors='none')
clean = df[['performance_score', 'salary']].dropna()
z  = np.polyfit(clean['performance_score'], clean['salary'], 1)
p  = np.poly1d(z)
xs = np.linspace(clean['performance_score'].min(), clean['performance_score'].max(), 200)
ax3.plot(xs, p(xs), color='#FDE74C', linewidth=2, linestyle='--', label='Trend')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x/1000:.0f}K'))
ax3.set_xlabel('Performance Score'); ax3.set_ylabel('Salary')
ax3.grid(color='#2A2D3E', linewidth=0.5)
ax3.legend(fontsize=8, framealpha=0.2, labelcolor=TEXT, facecolor=CARD, edgecolor='#2A2D3E')

# ── Panel 4: Heatmap — satisfaction by dept & seniority ───────────
ax4 = fig.add_subplot(gs[1, 2])
style_ax(ax4, '😊 Avg Satisfaction')
pivot = df.pivot_table('satisfaction_score', index='department',
                        columns='seniority', aggfunc='mean')
sns.heatmap(pivot, ax=ax4, cmap='RdYlGn', annot=True, fmt='.1f',
            linewidths=0.5, linecolor=BG, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 9, 'color': BG})
ax4.set_xlabel('Seniority'); ax4.set_ylabel('')
ax4.tick_params(axis='x', rotation=30, labelsize=8)
ax4.tick_params(axis='y', rotation=0,  labelsize=8, colors=TEXT)

# ── Panel 5: Hexbin — experience vs performance ────────────────────
ax5 = fig.add_subplot(gs[2, 0])
style_ax(ax5, '🧠 Experience vs Performance')
hb = ax5.hexbin(df['experience_years'], df['performance_score'],
                gridsize=18, cmap='plasma', mincnt=1)
cb = plt.colorbar(hb, ax=ax5, pad=0.01)
cb.ax.yaxis.label.set_color(TEXT); cb.ax.tick_params(colors=TEXT)
ax5.set_xlabel('Experience (yrs)'); ax5.set_ylabel('Performance Score')
ax5.grid(color='#2A2D3E', linewidth=0.4)

# ── Panel 6: Histogram — projects completed ────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
style_ax(ax6, '📦 Projects Completed')
ax6.hist(df['projects_completed'], bins=20, color=ACC, edgecolor=BG, alpha=0.85)
ax6.axvline(df['projects_completed'].mean(), color='#FDE74C', linewidth=2,
            linestyle='--', label=f"Mean: {df['projects_completed'].mean():.1f}")
ax6.legend(fontsize=9, labelcolor=TEXT, facecolor=CARD, edgecolor='#2A2D3E', framealpha=0.4)
ax6.set_xlabel('Projects Completed'); ax6.set_ylabel('Count')
ax6.grid(axis='y', color='#2A2D3E', linewidth=0.5)

# ── Panel 7: Stacked bar — salary band × seniority ────────────────
ax7 = fig.add_subplot(gs[2, 2])
style_ax(ax7, '🎯 Salary Band × Seniority')
band_colors = {'Low': '#FF6584', 'Mid': '#F7A072', 'High': '#43B89C', 'Premium': '#6C63FF'}
sb     = df.groupby(['seniority', 'salary_band']).size().unstack(fill_value=0)
sb_pct = sb.div(sb.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(sb_pct))
for band in ['Low', 'Mid', 'High', 'Premium']:
    if band in sb_pct.columns:
        vals = sb_pct[band].values
        ax7.bar(sb_pct.index, vals, bottom=bottom, color=band_colors[band],
                label=band, edgecolor=BG, linewidth=0.5)
        bottom += vals
ax7.set_ylabel('%'); ax7.set_ylim(0, 100)
ax7.legend(fontsize=8, labelcolor=TEXT, facecolor=CARD, edgecolor='#2A2D3E',
           framealpha=0.4, loc='lower right')
ax7.grid(axis='y', color='#2A2D3E', linewidth=0.5)
ax7.tick_params(axis='x', rotation=20)

# ── Panel 8: Horizontal bar — city avg salary ──────────────────────
ax8 = fig.add_subplot(gs[3, :2])
style_ax(ax8, '🏙️ Average Salary by City')
city_sal    = df.groupby('city')['salary'].mean().sort_values()
colors_city = plt.cm.cool(np.linspace(0.3, 0.9, len(city_sal)))
bars = ax8.barh(city_sal.index, city_sal.values, color=colors_city, edgecolor=BG, height=0.55)
for bar, val in zip(bars, city_sal.values):
    ax8.text(val + 300, bar.get_y() + bar.get_height() / 2,
             f'₹{val/1000:.1f}K', va='center', fontsize=9, color=TEXT)
ax8.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{x/1000:.0f}K'))
ax8.set_xlabel('Avg Salary'); ax8.set_xlim(0, city_sal.max() * 1.14)
ax8.grid(axis='x', color='#2A2D3E', linewidth=0.5)
ax8.tick_params(axis='y', colors=TEXT)

# ── Panel 9: KPI summary cards ─────────────────────────────────────
ax9 = fig.add_subplot(gs[3, 2])
ax9.set_facecolor(CARD); ax9.set_xlim(0, 1); ax9.set_ylim(0, 1)
for sp in ax9.spines.values(): sp.set_visible(False)
ax9.set_xticks([]); ax9.set_yticks([])
ax9.set_title('📊 Key Stats', pad=10, color=TEXT, fontsize=12, fontweight='bold')
kpis = [
    ('Total Employees',  f'{len(df):,}',                          '#6C63FF'),
    ('Avg Salary',       f'₹{df["salary"].mean()/1000:.1f}K',     '#43B89C'),
    ('Avg Performance',  f'{df["performance_score"].mean():.1f}',  '#F7A072'),
    ('Avg Satisfaction', f'{df["satisfaction_score"].mean():.1f}/10','#5BC0EB'),
]
for i, (label, val, color) in enumerate(kpis):
    y = 0.82 - i * 0.22
    ax9.add_patch(mpatches.FancyBboxPatch(
        (0.05, y - 0.08), 0.9, 0.18, boxstyle="round,pad=0.02",
        facecolor=color + '22', edgecolor=color, linewidth=1.5))
    ax9.text(0.5, y + 0.02, val, ha='center', va='center',
             fontsize=16, fontweight='bold', color=color)
    ax9.text(0.5, y - 0.04, label, ha='center', va='center',
             fontsize=9, color='#9E9EC8')

# ── Save ───────────────────────────────────────────────────────────
OUTPUT = 'hr_analytics_dashboard.png'
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor=BG)
print(f"\nDashboard saved → {OUTPUT}")
