import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from arch import arch_model
import scipy.stats as stats
import plotly.figure_factory as ff


# Function to fetch historical data for individual stocks
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Adj Close'].pct_change().dropna()
    return data

# Function to fetch market data (e.g., S&P 500)
def fetch_market_data(start_date, end_date):
    market_data = yf.download('^GSPC', start=start_date, end=end_date)
    market_data['Returns'] = market_data['Adj Close'].pct_change().dropna()
    return market_data

# Function to calculate beta
def calculate_beta(stock_returns, market_returns):
    cov_matrix = np.cov(stock_returns, market_returns)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    return beta

# Functions for different models
def jump_diffusion_model(S0, mu, sigma, lam, jump_mean, jump_std, T, N, simulations):
    dt = T / N
    size = (simulations, N)
    poisson_rv = np.random.poisson(lam * dt, size=size)
    normal_rv = np.random.normal(0, 1, size=size)
    jump_rv = np.random.normal(jump_mean, jump_std, size=size)
    returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * normal_rv + jump_rv * poisson_rv
    price_paths = S0 * np.exp(np.cumsum(returns, axis=1))
    return price_paths

def two_factor_model(S0, alpha, beta, market_returns, sigma_idiosyncratic, T, N, simulations):
    dt = T / N
    size = (simulations, N)
    market_mean = market_returns.mean()
    market_std = market_returns.std()
    simulated_market_returns = np.random.normal(market_mean * dt, market_std * np.sqrt(dt), size=size)
    idiosyncratic_returns = np.random.normal(0, sigma_idiosyncratic * np.sqrt(dt), size=size)
    returns = alpha * dt + beta * simulated_market_returns + idiosyncratic_returns
    price_paths = S0 * np.exp(np.cumsum(returns, axis=1))
    return price_paths

def estimate_garch_volatility(returns, forecast_horizon):
    returns = returns.dropna()
    returns = returns[np.isfinite(returns)]
    model = arch_model(returns, vol='GARCH', p=1, q=1)
    results = model.fit(disp='off')
    forecast = results.forecast(horizon=forecast_horizon)
    return np.sqrt(forecast.variance.values[-1, :])

def garch_simulation(S0, mu, garch_volatility, T, N, simulations):
    dt = T / N
    size = (simulations, N)
    returns = mu * dt + garch_volatility * np.sqrt(dt) * np.random.randn(simulations, N)
    price_paths = S0 * np.exp(np.cumsum(returns, axis=1))
    return price_paths

def realistic_simulation(S0, mu, sigma, T, N, simulations, trading_cost_pct, dividend_yield):
    dt = T / N
    size = (simulations, N)
    returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(simulations, N))
    price_paths = np.zeros((simulations, N + 1))
    price_paths[:, 0] = S0
    for t in range(1, N + 1):
        price_paths[:, t] = price_paths[:, t-1] * returns[:, t-1]
        price_paths[:, t] *= (1 - trading_cost_pct)
        price_paths[:, t] *= np.exp(dividend_yield * dt)
    return price_paths


# Function to calculate key metrics and risk metrics for individual stocks
def calculate_metrics(price_paths):
    final_prices = price_paths[:, -1]
    
    metrics = {
        'mean': np.mean(final_prices),
        'median': np.median(final_prices),
        'std_dev': np.std(final_prices),
        'skewness': stats.skew(final_prices),
        'kurtosis': stats.kurtosis(final_prices),
        'VaR_95': np.percentile(final_prices, 5),
        'VaR_99': np.percentile(final_prices, 1),
        'ES_95': np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)]),
        'ES_99': np.mean(final_prices[final_prices <= np.percentile(final_prices, 1)])
    }
    
    return metrics

def create_styled_metrics(metrics):
    df = pd.DataFrame(metrics, index=['Value'])
    df = df.T  # Transpose the dataframe
    
    # Define custom styles
    styles = [
        dict(selector="th", props=[("font-weight", "bold"), ("text-align", "left"), ("padding", "10px")]),
        dict(selector="td", props=[("text-align", "right"), ("padding", "10px")]),
        dict(selector="", props=[("border", "1px solid #ddd"), ("font-family", "Arial, sans-serif")])
    ]
    
    # Apply styles and format numbers
    styled_df = df.style.set_table_styles(styles)\
                  .format("{:.2f}")\
                  .set_properties(**{'background-color': '#78b52d', 'color': '#333'})
    
    return styled_df


def calculate_confidence_interval(price_paths, confidence=0.95):
    final_prices = price_paths[:, -1]
    lower = np.percentile(final_prices, (1 - confidence) / 2 * 100)
    upper = np.percentile(final_prices, (1 + confidence) / 2 * 100)
    return lower, upper

# Plotly visualization functions for individual stocks
def plot_simulation_plotly(price_paths, title):
    fig = go.Figure()
    for i in range(min(100, price_paths.shape[0])):  # Plot up to 100 paths
        fig.add_trace(go.Scatter(y=price_paths[i], mode='lines', opacity=0.1, line=dict(color='blue'), showlegend=False))
    fig.update_layout(title=title, xaxis_title='Days', yaxis_title='Price')
    return fig

# Function to fetch historical data for multiple stocks
def fetch_data_multiple(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    returns = data.pct_change().dropna()
    return data, returns

# Function to simulate portfolio returns
def simulate_portfolio(returns, weights, simulations, days):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Simulate returns using Cholesky decomposition
    L = np.linalg.cholesky(cov_matrix)
    
    # Generate correlated random returns
    rand_returns = np.random.normal(0, 1, size=(days, len(weights), simulations))
    correlated_returns = mean_returns.values.reshape(-1, 1, 1) + np.dot(L, rand_returns)
    
    # Calculate portfolio returns
    portfolio_returns = np.sum(correlated_returns * weights.reshape(-1, 1, 1), axis=0)
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + portfolio_returns, axis=0)
    
    return cumulative_returns

# Function to calculate portfolio metrics
def calculate_portfolio_metrics(final_values):
    metrics = {
        'mean': np.mean(final_values),
        'median': np.median(final_values),
        'std_dev': np.std(final_values),
        'skewness': stats.skew(final_values),
        'kurtosis': stats.kurtosis(final_values),
        'VaR_95': np.percentile(final_values, 5),
        'VaR_99': np.percentile(final_values, 1),
        'ES_95': np.mean(final_values[final_values <= np.percentile(final_values, 5)]),
        'ES_99': np.mean(final_values[final_values <= np.percentile(final_values, 1)])
    }
    return metrics

# Plotly visualization function for portfolio
def plot_portfolio_simulation(cumulative_returns, title):
    fig = go.Figure()
    for i in range(min(100, cumulative_returns.shape[1])):  # Plot up to 100 paths
        fig.add_trace(go.Scatter(y=cumulative_returns[:, i], mode='lines', opacity=0.1, line=dict(color='blue'), showlegend=False))
    fig.update_layout(title=title, xaxis_title='Days', yaxis_title='Portfolio Value')
    return fig

# Streamlit app
st.title('Monte Carlo Simulation for Stock Prices and Portfolio')

tickers = [
    'AAPL', 'ADBE', 'AMZN', 'CSCO', 
    'DAL', 'ETN', 'FDX', 
    'GE', 'GOOG', 'IBM', 
    'JNJ', 'LCID', 'META', 'MMM', 
    'MSEQX', 'MSFT', 'NFLX', 'RCL', 'RIVN', 'SPY', 'TSLA', 'V', 
    'WBD', 'WSM'
]

# Remove duplicates by converting the list to a set and then back to a list
stock_list = list(set(tickers))

# List of popular stocks
# stock_list = [
#     'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 
#     'TSLA', 'BRK-B', 'JPM', 'JNJ', 'V', 
#     'PG', 'UNH', 'HD', 'BAC', 'DIS'
# ]

# Dropdown for individual stock selection
ticker = st.selectbox('Select a stock for individual simulation', stock_list)

start_date = st.date_input('Start date', value=pd.to_datetime('2020-01-01'))
end_date = st.date_input('End date', value=pd.to_datetime('2023-01-01'))
forecast_years = st.slider('Number of years to forecast', 0.5, 5.0, 1.0)
simulations = st.slider('Number of simulation runs', 1000, 10000, 5000)

if st.button('Run Individual Stock Simulation'):
    data = fetch_data(ticker, start_date, end_date)
    market_data = fetch_market_data(start_date, end_date)

    historical_returns = data['Returns']
    market_returns = market_data['Returns']

    # Data cleaning
    historical_returns = historical_returns.dropna()
    historical_returns = historical_returns[np.isfinite(historical_returns)]
    market_returns = market_returns.dropna()
    market_returns = market_returns[np.isfinite(market_returns)]

    if len(historical_returns) == 0 or len(market_returns) == 0:
        st.write("Not enough valid data to run simulations.")
    else:
        S0 = data['Adj Close'][-1]
        mu = historical_returns.mean()
        sigma = historical_returns.std()
        beta = calculate_beta(historical_returns, market_returns)
        alpha = mu - beta * market_returns.mean()
        sigma_idiosyncratic = np.sqrt(historical_returns.var() - beta**2 * market_returns.var())
        lam = 10  # Average number of jumps per year
        jump_mean = 0
        jump_std = sigma * 2  # Larger jumps

        # Estimate GARCH volatility
        garch_volatility = estimate_garch_volatility(historical_returns, int(forecast_years * 252))

        T = forecast_years
        N = int(T * 252)

        # Run simulations
        price_paths_jump_diffusion = jump_diffusion_model(S0, mu, sigma, lam, jump_mean, jump_std, T, N, simulations)
        price_paths_two_factor = two_factor_model(S0, alpha, beta, market_returns, sigma_idiosyncratic, T, N, simulations)
        price_paths_garch = garch_simulation(S0, mu, garch_volatility, T, N, simulations)
        price_paths_realistic = realistic_simulation(S0, mu, sigma, T, N, simulations, 0.001, 0.02)  # 0.1% trading cost, 2% dividend yield

        # Calculate metrics and confidence intervals
        metrics_jump_diffusion = calculate_metrics(price_paths_jump_diffusion)
        metrics_two_factor = calculate_metrics(price_paths_two_factor)
        metrics_garch = calculate_metrics(price_paths_garch)
        metrics_realistic = calculate_metrics(price_paths_realistic)

        # Calculate confidence intervals for each model
        ci_jump_diffusion = calculate_confidence_interval(price_paths_jump_diffusion)
        ci_two_factor = calculate_confidence_interval(price_paths_two_factor)
        ci_garch = calculate_confidence_interval(price_paths_garch)
        ci_realistic = calculate_confidence_interval(price_paths_realistic)

        # Create columns for displaying the metrics and confidence intervals side by side
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Jump Diffusion")
            st.dataframe(create_styled_metrics(metrics_jump_diffusion))
            st.write(f"95% Confidence Interval: ({ci_jump_diffusion[0]:.2f}, {ci_jump_diffusion[1]:.2f})")

        with col2:
            st.subheader("Two-Factor Model")
            st.dataframe(create_styled_metrics(metrics_two_factor))
            st.write(f"95% Confidence Interval: ({ci_two_factor[0]:.2f}, {ci_two_factor[1]:.2f})")

        with col3:
            st.subheader("GARCH Model")
            st.dataframe(create_styled_metrics(metrics_garch))
            st.write(f"95% Confidence Interval: ({ci_garch[0]:.2f}, {ci_garch[1]:.2f})")

        with col4:
            st.subheader("Realistic Simulation")
            st.dataframe(create_styled_metrics(metrics_realistic))
            st.write(f"95% Confidence Interval: ({ci_realistic[0]:.2f}, {ci_realistic[1]:.2f})")

        # Plot results using Plotly
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(plot_simulation_plotly(price_paths_jump_diffusion, 'Jump Diffusion Model'))
            st.plotly_chart(plot_simulation_plotly(price_paths_two_factor, 'Two-Factor Model'))

        with col2:
            st.plotly_chart(plot_simulation_plotly(price_paths_garch, 'GARCH Model'))
            st.plotly_chart(plot_simulation_plotly(price_paths_realistic, 'Realistic Simulation'))

# Portfolio Simulation Section
st.header('Monte Carlo Simulation for Portfolio')

# Multiple selection for stocks
selected_stocks = st.multiselect('Select stocks for your portfolio', stock_list, default=['AAPL', 'MSFT'])

if len(selected_stocks) > 0:
    if st.button('Run Portfolio Simulation'):
        # Fetch historical data
        data, returns = fetch_data_multiple(selected_stocks, start_date, end_date)
        
        # Equal weights for simplicity (can be modified to allow user input)
        weights = np.array([1/len(selected_stocks)] * len(selected_stocks))
        
        # Run simulation
        days = int(forecast_years * 252)
        cumulative_returns = simulate_portfolio(returns, weights, simulations, days)
        
        # Calculate metrics
        final_values = cumulative_returns[-1, :] * 100000  # Assuming initial investment of $100,000
        metrics = calculate_portfolio_metrics(final_values)
        
        # Display results
        st.subheader("Portfolio Metrics")
        st.dataframe(create_styled_metrics(metrics))
        st.write(f"95% Confidence Interval: ({np.percentile(final_values, 2.5):.2f}, {np.percentile(final_values, 97.5):.2f})")
        
        # Plot results
        st.plotly_chart(plot_portfolio_simulation(cumulative_returns, 'Portfolio Simulation'))
        
        # Display correlation matrix
        corr_matrix = returns.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            zmin=-1, zmax=1,
            colorscale='RdBu_r',
            colorbar=dict(title='Correlation'),
            hoverongaps=False
        ))

        # Add text annotations
        annotations = []
        for i, row in enumerate(corr_matrix.values):
            for j, value in enumerate(row):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f'{value:.2f}',
                        font=dict(color='white' if abs(value) > 0.5 else 'black'),
                        showarrow=False
                    )
                )

        # Update layout
        fig.update_layout(
            title='Correlation Matrix Heatmap',
            xaxis=dict(
                tickmode='array',
                tickvals=np.arange(len(selected_stocks)),
                ticktext=selected_stocks,
                side='top'
            ),
            yaxis=dict(
                tickmode='array',
                tickvals=np.arange(len(selected_stocks)),
                ticktext=selected_stocks
            ),
            annotations=annotations,
            height=600,
            width=800,
            margin=dict(l=100, r=50, t=100, b=50)
        )

        # Display the heatmap in Streamlit
        st.write("Correlation Matrix:")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Please select at least one stock for the portfolio.")
