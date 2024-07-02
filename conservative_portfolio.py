import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import seaborn as sns



# Function to create yield curve charts
def generate_color_palette(n_colors):
    return sns.color_palette("husl", n_colors)

def create_yield_curve_chart(maturities, yields, title, first_point_offset, x_labels=None, xlim=(0, 35), ylim_offset=1):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='lightgrey')
    ax.scatter(maturities, yields, color='dodgerblue', s=100, edgecolors='black', linewidth=1.5)

    # Create annotations
    for i, (x, y) in enumerate(zip(maturities, yields)):
        offset = first_point_offset if i == 0 else (10, 10) if i == 1 else (0, 10)
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
    if x_labels:
        ax.set_xticklabels(x_labels)
    ax.set_xlim(xlim)
    ax.set_ylim(0, max(yields) + ylim_offset)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey', alpha=0.7)
    ax.legend(fontsize=12)

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    plt.figtext(0.99, 0.01, f'Updated: {timestamp}', horizontalalignment='right', fontsize=10, color='grey', fontstyle='italic')

    return fig

# Data and chart definitions
charts_data = [
    ([1, 2, 5, 10, 30], [2.039, 4.117, 4.252, 4.389, 4.55], 'U.S. Treasury Yield Curve Current Market Prices', (30, -5)),
    ([1, 2, 5, 10, 30], [5.12, 4.78, 4.44, 4.48, 4.65], 'U.S. Treasury Yield Curve', (0, 10)),
    ([1/12, 3/12, 6/12, 12/12], [5.359, 5.371, 5.309, 5.130], 'Short-Term Treasury Yield Curve', (0, 10), ['1 Month', '3 Months', '6 Months', '12 Months'], (0, 1.1), 0.5),
    ([1, 2, 5, 10, 30], [3.487, 3.529, 3.309, 3.516, 4.122], 'California Municipal Bond Yield Curve', (5, -18)),
    ([1, 2, 5, 10, 25], [2.788, 3.593, 4.716, 4.412, 5.004], 'California Municipal Yield Curve Current Market Prices', (30, 0), None, (0, 30)),
    ([1, 2, 5, 10, 17], [1.967, 2.166, 2.701, 2.835, 3.981], 'Triple Tax-Exempt California Municipal Yield Curve Current Market Prices', (0, 10), None, (0, 20)),
    ([1, 2, 5, 10, 30], [3.2, 3.15, 2.98, 2.89, 3.83], 'National Municipal Bond Yield Curve', (-5, 10))
]

# Streamlit app setup with improved UI
st.set_page_config(page_title="Conservative Portfolio Guide", page_icon="📊", layout="wide")

# Custom CSS to improve the look and feel
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle, .stHeader {
        color: #2c3e50;
    }
    .stMarkdown {
        font-size: 18px;
        line-height: 1.6;
    }
    .stButton>button {
        color: #ffffff;
        background-color: #3498db;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Steps", "Sample Portfolio", "Yield Curves"])

# Main content
st.title("Conservative Portfolio Selection")

if page == "Introduction":
    st.write("### Creating a Conservative Portfolio: A Step-by-Step Guide")
   # st.image("https://example.com/conservative_portfolio_image.jpg", caption="Conservative Portfolio Strategy")
    st.write("Welcome to our guide on creating a conservative investment portfolio. This application will walk you through the key steps and considerations for building a portfolio designed for stability and steady growth.")

elif page == "Steps":
    st.header("Step 1: Evaluate Inflation – \"Don't Fight the Fed\"")
    with st.expander("Read more about evaluating inflation"):
        st.write("""
        Inflation is a critical factor to consider when constructing a conservative portfolio. High inflation
        erodes the purchasing power of money, which can negatively impact real returns on investments.
        To navigate this, it's essential to keep an eye on the Federal Reserve's (Fed) actions and policies.

        The Federal Reserve plays a pivotal role in managing inflation through monetary policy. By
        adjusting interest rates and utilizing other tools, the Fed aims to keep inflation within a target
        range. The phrase "Don't fight the Fed" emphasizes the importance of aligning your investment
        strategy with the Fed's policy direction.

        - Regularly review inflation reports such as the Consumer Price Index (CPI) and Producer
          Price Index (PPI). These indices provide insights into the rate of inflation experienced by
          consumers and producers, respectively.
        - Stay updated on the Fed's announcements regarding interest rate changes and other
          monetary policy measures. These actions can signal shifts in inflation expectations.
        - Consider including Treasury Inflation-Protected Securities (TIPS) in your portfolio. TIPS
          are designed to provide protection against inflation by adjusting the principal value based
          on changes in the CPI.

        **Example**: In 2023, the U.S. experienced higher inflation rates due to supply chain disruptions
        and increased consumer demand post-pandemic. The Fed responded by raising interest rates
        multiple times. Investors who monitored these developments and adjusted their portfolios
        accordingly, perhaps by increasing holdings in TIPS or reducing exposure to long-term bonds,
        were better positioned to manage inflation risk.
        """)

    st.header("Step 2: Analyze the Money Supply and Predict Interest Rates")
    with st.expander("Read more about analyzing money supply"):
        st.write("""
        The money supply and interest rates are closely linked, influencing economic activity and
        investment returns. Understanding these elements helps in predicting future market conditions
        and making informed investment decisions.

        - Track indicators such as the M2 money supply, which includes cash, checking deposits,
          and easily convertible near money. An increase in the money supply can lead to higher
          inflation and interest rates.
        - Monitor the federal funds rate, the interest rate at which depository institutions lend
          balances to each other overnight. Changes in this rate can influence other interest rates,
          including those for mortgages, loans, and savings.
        - Analyze the yield curve, which plots interest rates of bonds with equal credit quality but
          differing maturity dates. A steepening yield curve can indicate rising interest rates, while
          a flattening or inverted yield curve may signal lower future rates.

        **Example**: In early 2023, the Fed increased the federal funds rate in response to persistent
        inflation, leading to a rise in short-term interest rates. Investors who analyzed these changes and
        adjusted their bond portfolios, perhaps by shifting to shorter-duration bonds to reduce interest
        rate risk, were able to better manage the impact of rising rates on their portfolios.
        """)

    st.header("Step 3: Examine the History of Interest Rates and the Current Yield Curves")
    with st.expander("Read more about examining interest rates"):
        st.write("""
        Interest rates and yield curves provide valuable insights into the overall economic environment
        and future expectations. Analyzing historical trends and the current state of these indicators can
        help in making informed investment decisions.

        - **Historical Trends**: Analyze historical data on interest rates to identify long-term trends.
          The Federal Reserve provides historical interest rate data that can be used for this
          analysis.
        - **Yield Curve Shapes**: Understand the different shapes of the yield curve:
          - **Normal Yield Curve**: Upward sloping, indicating that long-term rates are higher
            than short-term rates, reflecting expectations of economic growth and moderate
            inflation.
          - **Steep Yield Curve**: Indicates expectations of higher future inflation and
            economic growth.
          - **Flat or Inverted Yield Curve**: Can signal economic slowdown or recession, as
            short-term rates are similar to or higher than long-term rates.
        - **Current Data**: Use real-time data from reliable sources such as the U.S. Department of
          the Treasury to examine the current yield curve. This data provides insights into market
          sentiment and future interest rate movements.

        **Example**: In mid-2023, the yield curve showed signs of flattening, indicating potential concerns
        about economic growth. Investors who monitored this data might have adjusted their portfolios
        by reducing exposure to long-term bonds, which are more sensitive to interest rate changes, and
        increasing exposure to short-term bonds or cash equivalents.
        """)

    st.header("Step 4: Determine Stock to Bond Allocation")
    with st.expander("Read more about stock to bond allocation"):
        st.write("""
        A key decision in constructing a conservative portfolio is the allocation between stocks and
        bonds. Traditional conservative portfolios often follow a 60/40 allocation, with 60% in stocks
        and 40% in bonds. However, different allocations can be effective depending on the investor's
        risk tolerance and market conditions.

        - Assess your risk tolerance and investment horizon. Conservative investors typically
          prioritize capital preservation and steady income over high returns.
        - Consider the current market environment. In a low-interest-rate environment, bonds
          might offer lower yields, leading to a potential shift towards dividend-paying stocks or
          alternative income-generating assets.
        - Diversify within asset classes. For example, within the stock allocation, include a mix of
          large-cap, mid-cap, and international stocks. For bonds, consider a mix of government,
          corporate, and municipal bonds.

        **Example**: In 2023, with rising interest rates and market volatility, some conservative investors
        adjusted their portfolios to a 50/50 allocation. They increased their exposure to high-quality,
        short-term bonds to reduce interest rate risk and included dividend-paying stocks to maintain
        income levels.
        """)

    st.header("Step 5: Assess Diversification Needs")
    with st.expander("Read more about diversification"):
        st.write("""
        Diversification is a crucial strategy in risk management. It involves spreading investments across
        various asset classes, sectors, and geographical regions to reduce the impact of any single
        investment's poor performance on the overall portfolio.

        - Diversify across different asset classes, including stocks, bonds, real estate, and
          commodities. Each asset class reacts differently to economic changes, which can help
          stabilize your portfolio.
        - Within the stock allocation, ensure exposure to various sectors such as technology,
          healthcare, consumer goods, and utilities. This reduces sector-specific risks.
        - Include international investments to benefit from growth in different regions and reduce
          exposure to country-specific risks.

        **Example**: A conservative portfolio might include a mix of U.S. and international bonds,
        large-cap and small-cap stocks, and some exposure to real estate investment trusts (REITs) and
        commodities like gold. This diversification helps cushion the portfolio against volatility in any
        single market or sector.
        """)

    st.header("Step 6: Implement Risk Management Strategies")
    with st.expander("Read more about risk management"):
        st.write("""
        Risk management is essential in maintaining a conservative portfolio. Strategies such as regular
        portfolio rebalancing, using stop-loss orders, and hedging can help manage risk and protect
        capital.

        - Periodically review and adjust your portfolio to maintain the desired asset allocation.
          This involves selling overperforming assets and buying underperforming ones to
          maintain balance.
        - Implement stop-loss orders to limit potential losses. This involves setting a predetermined
          price at which an asset will be sold if its value falls to that level.
        - Consider hedging strategies such as options or futures to protect against downside risks.
          For example, using put options can provide insurance against significant declines in stock
          prices.

        **Example**: An investor might set a stop-loss order at 10% below the purchase price of a stock. If
        the stock's price drops to that level, it will be sold automatically, limiting the loss. Additionally,
        the investor might use put options on a portion of their stock holdings to hedge against market
        downturns.
        """)

elif page == "Sample Portfolio":
    st.header("Select Stocks and Bonds for the Portfolio")
    with st.expander("Read more about selecting stocks and bonds"):
        st.write("""
        The final step involves selecting specific stocks and bonds that align with your conservative
        investment strategy. This requires careful analysis and consideration of various factors to ensure
        the chosen investments meet your risk tolerance and financial goals.

        **Stock Selection:**
        - Blue-Chip Stocks: Focus on established companies with strong financials and a history
          of stable earnings and dividends. Examples include Johnson & Johnson, Procter & Gamble, and Microsoft.
        - Dividend-Paying Stocks: Choose companies that have a history of paying regular
          dividends. These stocks can provide a steady income stream, which is valuable in a
          conservative portfolio.
        - Low-Volatility Stocks: Look for stocks with lower volatility compared to the broader
          market. These stocks tend to be less affected by market swings, providing more stability.

        **Bond Selection:**
        - Government Bonds: U.S. Treasury bonds are considered one of the safest investments.
          They are backed by the full faith and credit of the U.S. government.
        - Investment-Grade Corporate Bonds: Select bonds from companies with high credit
          ratings (BBB or higher). These bonds offer a balance of safety and higher yields
          compared to government bonds.
        - Municipal Bonds: Consider municipal bonds, which provide tax-free interest income
          and are generally considered low risk.

        **Example Portfolio:**
        - Stocks: 20% in blue-chip stocks like Johnson & Johnson and Procter & Gamble, 20% in
          dividend-paying stocks like AT&T and Coca-Cola, 20% in low-volatility stocks like
          utilities and consumer staples.
        - Bonds: 30% in U.S. Treasury bonds, 20% in investment-grade corporate bonds from
          companies like Apple and General Electric, 10% in municipal bonds.

        By carefully selecting high-quality stocks and bonds, you can create a conservative portfolio designed to provide stability, income, and modest growth.
        """)

    # Interactive portfolio builder
    st.subheader("Build Your Conservative Portfolio")
    stocks = st.slider("Stock Allocation (%)", 0, 100, 60)
    bonds = 100 - stocks
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Stocks", f"{stocks}%")
    with col2:
        st.metric("Bonds", f"{bonds}%")
    
    if st.button("Generate Sample Portfolio"):
        st.write(f"Based on a {stocks}/{bonds} stock/bond split, here's a sample portfolio:")
        
        # Define sample investments
        stock_investments = [
            ("US Large Cap Stocks", 0.5),
            ("US Mid Cap Stocks", 0.2),
            ("International Developed Stocks", 0.2),
            ("Emerging Market Stocks", 0.1)
        ]
        
        bond_investments = [
            ("US Treasury Bonds", 0.4),
            ("Corporate Bonds", 0.3),
            ("Municipal Bonds", 0.2),
            ("International Bonds", 0.1)
        ]
        
        # Generate portfolio
        portfolio = []
        for name, weight in stock_investments:
            allocation = round(stocks * weight, 2)
            portfolio.append({"Asset": name, "Type": "Stock", "Allocation (%)": allocation})
        
        for name, weight in bond_investments:
            allocation = round(bonds * weight, 2)
            portfolio.append({"Asset": name, "Type": "Bond", "Allocation (%)": allocation})
        
        # Create a DataFrame
        df = pd.DataFrame(portfolio)
        
        # Display the portfolio
        st.dataframe(df.style.format({"Allocation (%)": "{:.2f}%"}))
    
    # Create a pie chart
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='#F0F2F6')
        colors = generate_color_palette(len(df))
        
        wedges, texts, autotexts = ax.pie(
            df["Allocation (%)"], 
            labels=df["Asset"], 
            autopct='%1.1f%%',
            startangle=90,
            colors=colors,
            wedgeprops=dict(width=0.6, edgecolor='white'),
            textprops=dict(color="black", fontsize=10, fontweight="bold")
        )

        # Add a circle at the center to make it a donut chart
        centre_circle = plt.Circle((0, 0), 0.3, fc='#F0F2F6')
        ax.add_artist(centre_circle)
        
        # Customize the appearance
        plt.setp(autotexts, size=9, weight="bold", color="white")
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

        # Add a title
        plt.title("Sample Portfolio Allocation", fontsize=16, fontweight="bold", pad=20)

        # Add a legend
        ax.legend(wedges, df["Asset"],
                title="Assets",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1))

        plt.tight_layout()
        st.pyplot(fig)
    
    # Display some advice
        st.info("""
        This is a sample portfolio based on your stock/bond allocation. Remember:
        1. Diversification is key to managing risk.
        2. Regular rebalancing helps maintain your desired allocation.
        3. Consider your personal financial situation and risk tolerance when investing.
        4. Consult with a financial advisor for personalized advice.
        """)

elif page == "Yield Curves":
    st.header("Yield Curve Charts")
    chart_selection = st.selectbox("Select a yield curve to display:", [data[2] for data in charts_data])
    
    for data in charts_data:
        if data[2] == chart_selection:
            fig = create_yield_curve_chart(*data)
            st.pyplot(fig)
            break

# Footer
st.markdown("---")
st.write("© 2024 Conservative Portfolio Guide. All rights reserved.")
