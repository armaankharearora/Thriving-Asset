import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import altair as alt

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
        st.metric("Actual Return", f"{client_data['Actual Return ']:.2f}%")
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Asset Allocation for {client}', fontsize=16)

    ax1.pie(asset_allocation.values(), labels=asset_allocation.keys(), autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
    ax1.set_title('Pie Chart')

    ax2.bar(asset_allocation.keys(), asset_allocation.values(), color=sns.color_palette("Set2"))
    ax2.set_title('Bar Chart')
    ax2.set_ylabel('Percentage')
    for i, v in enumerate(asset_allocation.values()):
        ax2.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

    st.pyplot(fig)

    st.subheader("Performance Comparison")

    performance_data = {
        'Actual Return': client_data['Actual Return '],
        'Expected Return': client_data['Expected Return '],
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

    st.subheader("All Client Data")
    st.write(client_data.to_frame().T)