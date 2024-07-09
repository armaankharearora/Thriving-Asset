import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import altair as alt
import pandas as pd

def create_scatter_plot(data):
    st.header('Scatter Plot Analysis')

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox('Select X-axis', numeric_columns, key='x_axis')
    with col2:
        y_axis = st.selectbox('Select Y-axis', numeric_columns, index=1, key='y_axis')
    with col3:
        color_options = ['Primary Owner'] + numeric_columns
        color_by = st.selectbox('Color by', color_options, key='color_by')

    # Create a new DataFrame with only the columns we need
    plot_data = data[['Primary Owner', x_axis, y_axis]].copy()
    if color_by != 'Primary Owner':
        plot_data[color_by] = data[color_by]

    # Calculate correlation
    corr = data[x_axis].corr(data[y_axis])

    st.subheader(f'{x_axis} vs {y_axis}')
    st.markdown(f"**Correlation:** {corr:.2f}")

    # Create the scatter plot using Altair
    chart = alt.Chart(plot_data).mark_circle(size=60).encode(
        x=alt.X(x_axis, title=x_axis),
        y=alt.Y(y_axis, title=y_axis),
        color=alt.Color(color_by, title=color_by),
        tooltip=['Primary Owner', x_axis, y_axis, color_by]
    ).interactive()



    st.altair_chart(chart, use_container_width=True)

  #  st.altair_chart(chart, use_container_width=True)

    # Add some explanatory text
    st.markdown("""
    This scatter plot shows the relationship between the selected variables. 
    Each point represents a client, and the color indicates the selected attribute.
    You can hover over points to see details and zoom in/out using the mouse wheel.
    """)


def create_bar_plot(data):
    st.subheader('Bar Chart')
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    x_axis = st.selectbox('Select metric for Bar Chart', numeric_columns)
    
    sorted_data = data.sort_values(by=x_axis, ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = sns.barplot(x='Primary Owner', y=x_axis, data=sorted_data, palette='viridis', ax=ax)
    
    plt.title(f'Top 15 Primary Owners by {x_axis}', fontsize=20, pad=20)
    plt.xlabel('Primary Owner', fontsize=14)
    plt.ylabel(x_axis, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    for i, v in enumerate(sorted_data[x_axis]):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)

def create_heatmap(data):
    st.subheader('Correlation Heatmap')
    
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    selected_columns = st.multiselect('Select columns for correlation analysis:', numeric_columns, default=numeric_columns[:8])
    
    if selected_columns:
        corr = data[selected_columns].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
        
        plt.title('Correlation Heatmap', fontsize=20, pad=20)
        plt.tight_layout()
        
        st.pyplot(fig)
    else:
        st.warning('Please select at least two columns for correlation analysis.')

def display_client_analysis(data):
    st.header('🧑‍💼 Individual Client Analysis')

    client = st.selectbox('Select a client:', data['Primary Owner'].unique())
    client_data = data[data['Primary Owner'] == client].iloc[0]

    st.subheader(f"Client: {client}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Actual Return", f"{client_data['Actual Return ']:.2f}%", 
                  delta=f"{client_data['Actual Return '] - client_data['Expected Return ']:.2f}%")
    with col2:
        st.metric("Expected Return", f"{client_data['Expected Return ']:.2f}%")
    with col3:
        st.metric("S&P", f"{client_data['S&P']:.2f}%")
    with col4:
        st.metric("Drift", f"{client_data['Drift']:.2f}%")
    with col5:
        st.metric("Sharpe Ratio", f"{client_data['Sharpe Ratio']:.2f}")

    st.subheader("Asset Allocation")
    
    asset_allocation = {
        'Equity': client_data['Equity %'],
        'Fixed Income': client_data['Fixed Income %'],
        'Cash': client_data['Cash %']
    }

    # Create a DataFrame for the asset allocation
    asset_df = pd.DataFrame(list(asset_allocation.items()), columns=['Asset', 'Percentage'])

    # Pie chart using Altair
    pie_chart = alt.Chart(asset_df).mark_arc().encode(
        theta='Percentage',
        color='Asset',
        tooltip=['Asset', 'Percentage']
    ).properties(
        title='Asset Allocation',
        width=300,
        height=300
    )

    # Bar chart using Altair
    bar_chart = alt.Chart(asset_df).mark_bar().encode(
        x='Asset',
        y='Percentage',
        color='Asset',
        tooltip=['Asset', 'Percentage']
    ).properties(
        title='Asset Allocation',
        width=300,
        height=300
    )

    # Display charts side by side
    st.altair_chart(alt.hconcat(pie_chart, bar_chart))

    st.subheader("Performance Comparison")

    performance_data = {
        'Metric': ['Actual Return', 'Expected Return', 'S&P'],
        'Value': [client_data['Actual Return '], client_data['Expected Return '], client_data['S&P']]
    }
    perf_df = pd.DataFrame(performance_data)

    chart = alt.Chart(perf_df).mark_bar().encode(
        x='Metric',
        y='Value',
        color='Metric',
        tooltip=['Metric', 'Value']
    ).properties(
        title='Performance Comparison',
        width=600,
        height=400
    )

    text = chart.mark_text(
        align='center',
        baseline='bottom',
        dy=-5
    ).encode(
        text=alt.Text('Value:Q', format='.2f')
    )

    st.altair_chart(chart + text)

    st.subheader("All Client Data")
    
    # Convert the series to a dataframe and transpose it
    client_df = client_data.to_frame().T
    
    # Function to color numeric cells based on their value
    def color_numeric(val):
        try:
            val = float(val)
            color = f'background-color: rgba(0, 255, 0, {val/100})' if val >= 0 else f'background-color: rgba(255, 0, 0, {-val/100})'
            return color
        except:
            return ''

    # Apply the styling
    styled_df = client_df.style.applymap(color_numeric)

    # Display the styled dataframe
    st.dataframe(styled_df)