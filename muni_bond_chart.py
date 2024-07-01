import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime


def create_yield_curve_chart(maturities, yields, title, first_point_offset):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='lightgrey')
    ax.scatter(maturities, yields, color='dodgerblue', s=100, edgecolors='black', linewidth=1.5)

    # Create annotations
    for i, (x, y) in enumerate(zip(maturities, yields)):
        if i == 0:
            offset = first_point_offset
        else:
            offset = (10, 10) if i == 1 else (0, 10)
        ax.annotate(
            f'{y}%', 
            (x, y), 
            textcoords="offset points", 
            xytext=offset, 
            ha='center', 
            fontsize=10, 
            fontweight='bold'
        )

    ax.plot(maturities, yields, color='green', linestyle='-', linewidth=2, marker='o', markerfacecolor='yellow', markersize=8, label='Yields')

    ax.set_xlabel('Bond Maturities (Years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Yield Rates (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(maturities)
    ax.set_xlim(0, 35)
    ax.set_ylim(0, max(yields) + 1)
    ax.set_yticks(range(0, int(max(yields) + 2)))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
    ax.legend(fontsize=12)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plt.figtext(0.99, 0.01, f'Updated: {timestamp}', horizontalalignment='right', fontsize=10, color='grey', fontstyle='italic')

    return fig

# Data for the first chart
maturities1 = [1, 2, 5, 10, 30]
yields1 = [2.039, 4.117, 4.252, 4.389, 4.55]
fig1 = create_yield_curve_chart(maturities1, yields1, 'U.S. Treasury Yield Curve Current Market Prices', first_point_offset=(30, -5))
st.pyplot(fig1)

# Data for the second chart
maturities2 = [1, 2, 5, 10, 30]
yields2 = [5.12, 4.78, 4.44, 4.48, 4.65]
fig2 = create_yield_curve_chart(maturities2, yields2, 'U.S. Treasury Yield Curve', first_point_offset=(0, 10))
st.pyplot(fig2)

short_term_maturities = [1/12, 3/12, 6/12, 12/12]  # Converting months to years
short_term_yields = [5.359, 5.371, 5.309, 5.130]

fig3, ax3 = plt.subplots(figsize=(12, 6), facecolor='lightgrey')
ax3.scatter(short_term_maturities, short_term_yields, color='dodgerblue', s=100, edgecolors='black', linewidth=1.5)

# Create annotations
for i, (x, y) in enumerate(zip(short_term_maturities, short_term_yields)):
    offset = (10, 10) if i == 1 else (0, 10)
    ax3.annotate(
        f'{y}%', 
        (x, y), 
        textcoords="offset points", 
        xytext=offset, 
        ha='center', 
        fontsize=10, 
        fontweight='bold'
    )

ax3.plot(short_term_maturities, short_term_yields, color='green', linestyle='-', linewidth=2, marker='o', markerfacecolor='yellow', markersize=8, label='Yields')

ax3.set_xlabel('Bond Maturities (Years)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Yield Rates (%)', fontsize=12, fontweight='bold')
ax3.set_title('Short-Term Treasury Yield Curve', fontsize=16, fontweight='bold')
ax3.set_xticks(short_term_maturities)
ax3.set_xticklabels(['1 Month', '3 Months', '6 Months', '12 Months'])
ax3.set_xlim(0, 1.1)
ax3.set_ylim(5, max(short_term_yields) + .25 )
ax3.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
ax3.legend(fontsize=12)

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
plt.figtext(0.99, 0.01, f'Updated: {timestamp}', horizontalalignment='right', fontsize=10, color='grey', fontstyle='italic')
st.pyplot(fig3)

# Data for the California municipal bond rates chart
california_maturities = [1, 2, 5, 10, 30]
california_yields = [3.487, 3.529, 3.309, 3.516, 4.122]

fig4, ax4 = plt.subplots(figsize=(12, 6), facecolor='lightgrey')
ax4.scatter(california_maturities, california_yields, color='dodgerblue', s=100, edgecolors='black', linewidth=1.5)

# Create annotations
annotations4 = [
    (1, 3.487, (0, 10)), 
    (2, 3.529, (10, 20)), 
    (5, 3.309, (0, 10)), 
    (10, 3.516, (0, 10)), 
    (30, 4.122, (0, 10))
]

for x, y, offset in annotations4:
    ax4.annotate(
        f'{y}%', 
        (x, y), 
        textcoords="offset points", 
        xytext=offset, 
        ha='center', 
        fontsize=10, 
        fontweight='bold'
    )

ax4.plot(california_maturities, california_yields, color='green', linestyle='-', linewidth=2, marker='o', markerfacecolor='yellow', markersize=8, label='Yields')

ax4.set_xlabel('Bond Maturities (Years)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Yield Rates (%)', fontsize=12, fontweight='bold')
ax4.set_title('California Municipal Bond Yield Curve', fontsize=16, fontweight='bold')
ax4.set_xticks(california_maturities)
ax4.set_xlim(0, 35)
ax4.set_ylim(0, max(california_yields) + 1)
ax4.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
ax4.legend(fontsize=12)

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
plt.figtext(0.99, 0.01, f'Updated: {timestamp}', horizontalalignment='right', fontsize=10, color='grey', fontstyle='italic')

st.pyplot(fig4)
# Data for the California live municipal bond YTW chart
california_live_muni_maturities = [1, 2, 5, 10, 25]
california_live_muni_ytw = [2.788, 3.593, 4.716, 4.412, 5.004]

fig5, ax5 = plt.subplots(figsize=(12, 6), facecolor='lightgrey')
ax5.scatter(california_live_muni_maturities, california_live_muni_ytw, color='dodgerblue', s=100, edgecolors='black', linewidth=1.5)

# Create annotations
annotations5 = [
    (1, 2.788, (30, -5)), 
    (2, 3.593, (0, 10)), 
    (5, 4.716, (0, 10)), 
    (10, 4.412, (0, 10)), 
    (25, 5.004, (0, 10))
]

for x, y, offset in annotations5:
    ax5.annotate(
        f'{y}%', 
        (x, y), 
        textcoords="offset points", 
        xytext=offset, 
        ha='center', 
        fontsize=10, 
        fontweight='bold'
    )

ax5.plot(california_live_muni_maturities, california_live_muni_ytw, color='green', linestyle='-', linewidth=2, marker='o', markerfacecolor='yellow', markersize=8, label='YTW')

ax5.set_xlabel('Bond Maturities (Years)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Yield to Worst (YTW) (%)', fontsize=12, fontweight='bold')
ax5.set_title('California Municipal Yield Curve Current Market Prices', fontsize=16, fontweight='bold')
ax5.set_xticks(california_live_muni_maturities)
ax5.set_xlim(0, 30)
ax5.set_ylim(0, max(california_live_muni_ytw) + 1)
ax5.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
ax5.legend(fontsize=12)

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
plt.figtext(0.99, 0.01, f'Updated: {timestamp}', horizontalalignment='right', fontsize=10, color='grey', fontstyle='italic')

st.pyplot(fig5)

# Data for the triple tax-exempt California live municipal bond YTW chart
triple_tax_exempt_maturities = [1, 2, 5, 10, 17]
triple_tax_exempt_ytw = [1.967, 2.166, 2.701, 2.835, 3.981]

fig6, ax6 = plt.subplots(figsize=(12, 6), facecolor='lightgrey')
ax6.scatter(triple_tax_exempt_maturities, triple_tax_exempt_ytw, color='dodgerblue', s=100, edgecolors='black', linewidth=1.5)

# Create annotations
annotations6 = [
    (1, 1.967, (0, 10)), 
    (2, 2.166, (0, 10)), 
    (5, 2.701, (0, 10)), 
    (10, 2.835, (0, 10)), 
    (17, 3.981, (0, 10))
]

for x, y, offset in annotations6:
    ax6.annotate(
        f'{y}%', 
        (x, y), 
        textcoords="offset points", 
        xytext=offset, 
        ha='center', 
        fontsize=10, 
        fontweight='bold'
    )

ax6.plot(triple_tax_exempt_maturities, triple_tax_exempt_ytw, color='green', linestyle='-', linewidth=2, marker='o', markerfacecolor='yellow', markersize=8, label='YTW')

ax6.set_xlabel('Bond Maturities (Years)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Yield to Worst (YTW) (%)', fontsize=12, fontweight='bold')
ax6.set_title('Triple Tax-Exempt California Municipal Yield Curve Current Market Prices', fontsize=16, fontweight='bold')
ax6.set_xticks(triple_tax_exempt_maturities)
ax6.set_xlim(0, 20)
ax6.set_ylim(0, max(triple_tax_exempt_ytw) + 1)
ax6.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
ax6.legend(fontsize=12)

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
plt.figtext(0.99, 0.01, f'Updated: {timestamp}', horizontalalignment='right', fontsize=10, color='grey', fontstyle='italic')

st.pyplot(fig6)


national_muni_maturities = [1, 2, 5, 10, 30]
national_muni_ytw = [3.2, 3.15, 2.98, 2.89, 3.83]

fig7, ax7 = plt.subplots(figsize=(12, 6), facecolor='lightgrey')
ax7.scatter(national_muni_maturities, national_muni_ytw, color='dodgerblue', s=100, edgecolors='black', linewidth=1.5)

# Create annotations
annotations7 = [
    (1, 3.2, (0, 10)), 
    (2, 3.15, (15, 10)), 
    (5, 2.98, (0, 10)), 
    (10, 2.89, (0, 10)), 
    (30, 3.83, (0, 10))
]

for x, y, offset in annotations7:
    ax7.annotate(
        f'{y}%', 
        (x, y), 
        textcoords="offset points", 
        xytext=offset, 
        ha='center', 
        fontsize=10, 
        fontweight='bold'
    )

ax7.plot(national_muni_maturities, national_muni_ytw, color='green', linestyle='-', linewidth=2, marker='o', markerfacecolor='yellow', markersize=8, label='YTW')

ax7.set_xlabel('Bond Maturities (Years)', fontsize=12, fontweight='bold')
ax7.set_ylabel('Yield to Worst (YTW) (%)', fontsize=12, fontweight='bold')
ax7.set_title('National Municipal Bond Yield Curve', fontsize=16, fontweight='bold')
ax7.set_xticks(national_muni_maturities)
ax7.set_xlim(0, 35)
ax7.set_ylim(0, max(national_muni_ytw) + 1)
ax7.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
ax7.legend(fontsize=12)

# Add timestamp to national municipal bond rates chart
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
plt.figtext(0.99, 0.01, f'Updated: {timestamp}', horizontalalignment='right', fontsize=10, color='grey', fontstyle='italic')

st.pyplot(fig7)