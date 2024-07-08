import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
file_path = 'Actual vs Expected Portofolio Performance July_2024 - Moderate.csv'
data = pd.read_csv(file_path)

risk_free_rate = 2.0  # 2%

# Calculate Sharpe Ratio for each client
data['Sharpe Ratio'] = (data['Actual Return '] - risk_free_rate) / data['Actual Return '].std()

# Set the title of the Streamlit app
st.set_page_config(page_title="Portfolio Performance Visualization", layout="wide")
st.title('📊 Portfolio Performance Visualization')

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Data Table', 'Visualizations', 'Summary Statistics', 'Client Analysis'])

# Sidebar for plot selection
if page == 'Visualizations':
    st.sidebar.title('Visualization Settings')
    plot_type = st.sidebar.selectbox('Select Plot Type', ['Scatter Plot', 'Bar Plot', 'Correlation Heatmap'])

# Data Table Page
if page == 'Data Table':
    st.subheader('Data Table')
    st.dataframe(data, width=1500, height=500)

# Visualizations Page
elif page == 'Visualizations':
    if plot_type == 'Scatter Plot':
        st.subheader('Scatter Plot')
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox('Select X-axis', data.columns[1:])
        with col2:
            y_axis = st.selectbox('Select Y-axis', data.columns[1:], index=1)
        
        # Create the scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = sns.scatterplot(data=data, x=x_axis, y=y_axis, hue='Primary Owner', 
                                  palette='viridis', s=100, alpha=0.7, ax=ax)
        
        # Customize the plot
        plt.title(f'{x_axis} vs {y_axis}', fontsize=20, fontweight='bold', pad=20)
        plt.xlabel(x_axis, fontsize=14, fontweight='bold')
        plt.ylabel(y_axis, fontsize=14, fontweight='bold')
        
        # Add a trend line
        x = data[x_axis]
        y = data[y_axis]
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
        
        # Add text for correlation coefficient
        corr = data[x_axis].corr(data[y_axis])
        plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax.transAxes, 
                 fontsize=12, verticalalignment='top')
        
        # Customize the legend
        plt.legend(title='Primary Owner', title_fontsize='13', fontsize='11', 
                   bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Improve tick labels
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Adjust layout and display the plot
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add description
        st.write(f"""
        This scatter plot shows the relationship between {x_axis} and {y_axis} for different primary owners.
        Each point represents a primary owner, with the color indicating the specific owner.
        The dashed red line represents the trend line, and the correlation coefficient is displayed in the top-left corner.
        """)
    elif plot_type == 'Correlation Heatmap':
        st.subheader('Correlation Heatmap')
        
        # Select columns for correlation analysis
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        selected_columns = st.multiselect('Select columns for correlation analysis:', numeric_columns, default=numeric_columns[:8])
        
        if selected_columns:
            corr = data[selected_columns].corr()
            
            # Create a mask to hide the upper triangle
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            # Set up the matplotlib figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create the heatmap
            sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                        square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
            
            # Customize the plot
            plt.title('Correlation Heatmap', fontsize=20, pad=20)
            plt.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
        else:
            st.warning('Please select at least two columns for correlation analysis.')
    elif plot_type == 'Bar Plot':
        st.subheader('Bar Chart')
        
        # Select column for bar chart
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        x_axis = st.selectbox('Select metric for Bar Chart', numeric_columns)
        
        # Sort data and get top 15 values
        sorted_data = data.sort_values(by=x_axis, ascending=False).head(15)
        
        # Create the bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = sns.barplot(x='Primary Owner', y=x_axis, data=sorted_data, palette='viridis', ax=ax)
        
        # Customize the plot
        plt.title(f'Top 15 Primary Owners by {x_axis}', fontsize=20, pad=20)
        plt.xlabel('Primary Owner', fontsize=14)
        plt.ylabel(x_axis, fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        
        # Add value labels on top of each bar
        for i, v in enumerate(sorted_data[x_axis]):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
        
        # Adjust layout and display the plot
        plt.tight_layout()
        st.pyplot(fig)

# Summary Statistics Page
elif page == 'Summary Statistics':
    st.header('📊 Summary Statistics')
    st.subheader('Average Sharpe Ratio')
    avg_sharpe_ratio = data['Sharpe Ratio'].mean()
    st.write(f"Average Sharpe Ratio: {avg_sharpe_ratio:.2f}")

    # Select columns for analysis
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    selected_columns = st.multiselect('Select columns for analysis:', numeric_columns, default=numeric_columns[:5])

    if selected_columns:
        # Calculate summary statistics
        summary = data[selected_columns].describe()

        # Round all values to 2 decimal places
        summary = summary.round(2)

        # Add additional statistics
        summary.loc['range'] = summary.loc['max'] - summary.loc['min']
        summary.loc['median'] = data[selected_columns].median()
        summary.loc['mode'] = data[selected_columns].mode().iloc[0]
        summary.loc['skew'] = data[selected_columns].skew()
        summary.loc['kurtosis'] = data[selected_columns].kurtosis()

        # Reorder rows
        summary = summary.reindex(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'range', 'median', 'mode', 'skew', 'kurtosis'])

        # Display the summary statistics
        st.subheader('Descriptive Statistics')
        st.dataframe(summary.style.highlight_max(axis=1, color='lightgreen').highlight_min(axis=1, color='lightpink'))

        # Data Distribution Visualizations
        st.subheader('Data Distribution')
        
        # Select a column for distribution analysis
        dist_column = st.selectbox('Select a column for distribution analysis:', selected_columns)

        col1, col2 = st.columns(2)

        with col1:
            # Histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data[dist_column], kde=True, ax=ax)
            plt.title(f'Distribution of {dist_column}', fontsize=16)
            plt.xlabel(dist_column, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            st.pyplot(fig)

        with col2:
            # Box plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(y=data[dist_column], ax=ax)
            plt.title(f'Box Plot of {dist_column}', fontsize=16)
            plt.ylabel(dist_column, fontsize=12)
            st.pyplot(fig)

        # Correlation analysis
        st.subheader('Correlation Analysis')
        corr_matrix = data[selected_columns].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        plt.title('Correlation Heatmap', fontsize=16)
        st.pyplot(fig)

        # Top correlations
        st.subheader('Top 5 Positive and Negative Correlations')
        corr_pairs = corr_matrix.unstack().sort_values(kind="quicksort").drop_duplicates()
        top_pos = corr_pairs[corr_pairs < 1].tail(5)
        top_neg = corr_pairs[corr_pairs < 1].head(5)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Top Positive Correlations:")
            st.dataframe(top_pos)
        with col2:
            st.write("Top Negative Correlations:")
            st.dataframe(top_neg)

    else:
        st.warning('Please select at least one column for analysis.')

elif page == 'Client Analysis':
    st.header('🧑‍💼 Individual Client Analysis')

    # Select a client
    client = st.selectbox('Select a client:', data['Primary Owner'].unique())

    # Filter data for the selected client
    client_data = data[data['Primary Owner'] == client].iloc[0]

    # Display client information
    st.subheader(f"Client: {client}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Actual Return", f"{client_data['Actual Return ']:.2f}%")  # Note the space after 'Return'
    with col2:
        st.metric("Expected Return", f"{client_data['Expected Return ']:.2f}%")  # Note the space after 'Return'
    with col3:
        st.metric("S&P", f"{client_data['S&P']:.2f}%")
    with col4:
        st.metric("Drift", f"{client_data['Drift']:.2f}%")
    with col5:
            st.metric("Sharpe Ratio", f"{client_data['Sharpe Ratio']:.2f}")

    # Asset Allocation
    st.subheader("Asset Allocation")
    
    asset_allocation = {
        'Equity': client_data['Equity %'],
        'Fixed Income': client_data['Fixed Income %'],
        'Cash': client_data['Cash %']
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Asset Allocation for {client}', fontsize=16)

    # Pie chart
    ax1.pie(asset_allocation.values(), labels=asset_allocation.keys(), autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
    ax1.set_title('Pie Chart')

    # Bar chart
    ax2.bar(asset_allocation.keys(), asset_allocation.values(), color=sns.color_palette("Set2"))
    ax2.set_title('Bar Chart')
    ax2.set_ylabel('Percentage')
    for i, v in enumerate(asset_allocation.values()):
        ax2.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

    st.pyplot(fig)

    # Performance Comparison
    st.subheader("Performance Comparison")

    performance_data = {
        'Actual Return': client_data['Actual Return '],  # Note the space after 'Return'
        'Expected Return': client_data['Expected Return '],  # Note the space after 'Return'
        'S&P': client_data['S&P']
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(performance_data.keys(), performance_data.values(), color=sns.color_palette("Set2"))
    ax.set_title('Performance Comparison')
    ax.set_ylabel('Percentage')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom')

    st.pyplot(fig)

    # Display all client data
    st.subheader("All Client Data")
    st.write(client_data.to_frame().T)
