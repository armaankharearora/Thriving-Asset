import streamlit as st
from data_loader import load_data
from visualizations import create_scatter_plot, create_bar_plot, create_heatmap, display_client_analysis
from utils import calculate_sharpe_ratio

# Load the data
data = load_data('Actual vs Expected Portofolio Performance July_2024 - Moderate.csv')

# Calculate Sharpe Ratio for each client
data['Sharpe Ratio'] = calculate_sharpe_ratio(data, 2.0)

# Set the title of the Streamlit app
#st.set_page_config(page_title="Portfolio Performance Visualization", layout="wide")
st.title('📊 Portfolio Performance Visualization')

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Data Table', 'Visualizations', 'Summary Statistics', 'Client Analysis'])

# Data Table Page
if page == 'Data Table':
    st.subheader('Data Table')
    st.dataframe(data, width=1500, height=500)

# Visualizations Page
elif page == 'Visualizations':
    st.sidebar.title('Visualization Settings')
    plot_type = st.sidebar.selectbox('Select Plot Type', ['Scatter Plot', 'Bar Plot', 'Correlation Heatmap'])

    if plot_type == 'Scatter Plot':
        create_scatter_plot(data)
    elif plot_type == 'Bar Plot':
        create_bar_plot(data)
    elif plot_type == 'Correlation Heatmap':
        create_heatmap(data)

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
        # Calculate and display summary statistics
        summary = data[selected_columns].describe().round(2)
        st.subheader('Descriptive Statistics')
        st.dataframe(summary.style.highlight_max(axis=1, color='lightgreen').highlight_min(axis=1, color='lightpink'))
    else:
        st.warning('Please select at least one column for analysis.')

# Client Analysis Page
elif page == 'Client Analysis':
    display_client_analysis(data)
