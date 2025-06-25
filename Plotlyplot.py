import re
from Config import call_groq_api
import json

def analyze_data_for_multiple_plots(result_df):
    """Analyze DataFrame to determine if multiple plots would be better"""
    numeric_cols = result_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = result_df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = result_df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Check for different axis types/scales
    multiple_plot_indicators = {
        'different_scales': False,
        'mixed_data_types': False,
        'too_many_categories': False,
        'suggested_plots': []
    }
    
    # Check if numeric columns have very different scales
    if len(numeric_cols) >= 2:
        scales = []
        for col in numeric_cols:
            col_range = result_df[col].max() - result_df[col].min()
            scales.append(col_range)
        
        # If scales differ by more than 2 orders of magnitude
        max_scale = max(scales)
        min_scale = min([s for s in scales if s > 0])
        if max_scale / min_scale > 100:
            multiple_plot_indicators['different_scales'] = True
    
    # Check for mixed data types that don't work well together
    if len(numeric_cols) > 0 and len(categorical_cols) > 0 and len(datetime_cols) > 0:
        multiple_plot_indicators['mixed_data_types'] = True
    
    # Check for too many categories
    for col in categorical_cols:
        if result_df[col].nunique() > 15:
            multiple_plot_indicators['too_many_categories'] = True
            break
    
    # Suggest plot types based on data
    if len(numeric_cols) >= 2 and len(categorical_cols) >= 1:
        multiple_plot_indicators['suggested_plots'].extend(['bar_chart', 'scatter_plot'])
    
    if len(datetime_cols) > 0 and len(numeric_cols) > 0:
        multiple_plot_indicators['suggested_plots'].append('time_series')
    
    if len(categorical_cols) > 0:
        multiple_plot_indicators['suggested_plots'].append('pie_chart')
    
    return multiple_plot_indicators

#----------------------------------------------------------------update--------------------------------------------------------------------------

def analyze_data_for_subplots(result_df):
    """Enhanced analysis to determine if multiple subplots would be beneficial"""
    numeric_cols = result_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = result_df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = result_df.select_dtypes(include=['datetime64']).columns.tolist()
    
    subplot_analysis = {
        'needs_subplots': False,
        'subplot_reasons': [],
        'suggested_layout': 'single',
        'plot_combinations': []
    }
    
    # Check for different scales
    if len(numeric_cols) >= 2:
        scales = []
        for col in numeric_cols:
            if result_df[col].notna().sum() > 0:  # Check for non-null values
                col_range = result_df[col].max() - result_df[col].min()
                if col_range > 0:
                    scales.append((col, col_range))
        
        if len(scales) >= 2:
            max_scale = max(scales, key=lambda x: x[1])[1]
            min_scale = min(scales, key=lambda x: x[1])[1]
            if min_scale > 0 and max_scale / min_scale > 100:
                subplot_analysis['needs_subplots'] = True
                subplot_analysis['subplot_reasons'].append('different_scales')
    
    # Check for multiple time series
    if len(datetime_cols) > 0 and len(numeric_cols) >= 2:
        subplot_analysis['needs_subplots'] = True
        subplot_analysis['subplot_reasons'].append('multiple_time_series')
        subplot_analysis['plot_combinations'].append({
            'type': 'time_series_subplots',
            'x_col': datetime_cols[0],
            'y_cols': numeric_cols[:4],  # Limit to 4 subplots
            'layout': 'vertical'
        })
    
    # Check for multiple categories to compare
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 2:
        unique_categories = result_df[categorical_cols[0]].nunique()
        if 3 <= unique_categories <= 8:  # Good range for subplots
            subplot_analysis['needs_subplots'] = True
            subplot_analysis['subplot_reasons'].append('category_comparison')
            subplot_analysis['plot_combinations'].append({
                'type': 'category_subplots',
                'category_col': categorical_cols[0],
                'numeric_cols': numeric_cols[:2],
                'layout': 'grid'
            })
    
    # Multiple metrics dashboard
    if len(numeric_cols) >= 4:
        subplot_analysis['needs_subplots'] = True
        subplot_analysis['subplot_reasons'].append('metrics_dashboard')
        subplot_analysis['plot_combinations'].append({
            'type': 'metrics_dashboard',
            'metrics': numeric_cols[:6],  # Limit to 6 metrics
            'layout': 'grid'
        })
    
    # Determine layout
    if subplot_analysis['needs_subplots']:
        num_plots = len(subplot_analysis['plot_combinations'])
        if num_plots <= 2:
            subplot_analysis['suggested_layout'] = 'horizontal'
        elif num_plots <= 4:
            subplot_analysis['suggested_layout'] = 'grid_2x2'
        else:
            subplot_analysis['suggested_layout'] = 'grid_2x3'
    
    return subplot_analysis

def generate_multiple_subplot_code(client, user_query, result_df, subplot_analysis, column_mapping):
    """Generate Plotly code for multiple subplots"""
    
    df_info = f"""
    DataFrame shape: {result_df.shape}
    Columns: {result_df.columns.tolist()}
    Data types: {result_df.dtypes.to_dict()}
    
    Column Details:
    {json.dumps([{'name': col, 'type': str(result_df[col].dtype), 'sample_values': result_df[col].dropna().head(3).tolist()} for col in result_df.columns], indent=2, default=str)}
    
    Subplot Analysis:
    {json.dumps(subplot_analysis, indent=2, default=str)}
    """
    
    # Determine the best subplot configuration
    plot_config = subplot_analysis['plot_combinations'][0] if subplot_analysis['plot_combinations'] else None
    
    if not plot_config:
        return None
    
    prompt = f"""
    Generate Python code using Plotly to create multiple subplots based on the analysis:
    
    User Query: "{user_query}"
    
    Data Information:
    {df_info}
    
    Subplot Configuration: {json.dumps(plot_config, indent=2)}
    
    Requirements:
    1. Use plotly.graph_objects and make_subplots for creating subplots
    2. The DataFrame variable name should be 'result_df'
    3. Use the EXACT column names as shown in the Columns list above
    4. Create appropriate subplot layout based on the configuration
    5. Return only the Python code between codestart and codeend markers
    6. Assign the figure to a variable called 'fig'
    7. Don't include fig.show() or any display commands
    8. Include proper titles and labels for each subplot
    9. Handle any potential data type issues
    10. Use appropriate chart types for each subplot
    
    Subplot Type Guidelines:
    - time_series_subplots: Create separate time series for each numeric column
    - category_subplots: Create subplots grouped by categories
    - metrics_dashboard: Create multiple metric visualizations
    
    Example format for time series subplots:
    codestart
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Plot 1', 'Plot 2', 'Plot 3', 'Plot 4']
    )
    
    # Add traces to subplots
    fig.add_trace(go.Scatter(x=result_df['x_col'], y=result_df['y_col1'], name='Series 1'), row=1, col=1)
    fig.add_trace(go.Scatter(x=result_df['x_col'], y=result_df['y_col2'], name='Series 2'), row=1, col=2)
    
    fig.update_layout(title_text="Multiple Subplots Dashboard")
    codeend
    
    Your response must begin with 'codestart' and end with 'codeend'
    """
    
    response = call_groq_api(client, prompt)
    if response:
        # Extract code between codestart and codeend markers
        match = re.search(r'codestart(.*?)codeend', response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            return code
        else:
            # Fallback extraction
            code = re.sub(r'```python\n?|```\n?|Python Code:\s*', '', response).strip()
            return code
    return None

def generate_enhanced_plotly_code(client, user_query, result_df, column_mapping):
    """Enhanced version that decides between single plot or subplots"""
    
    # First, analyze if subplots would be beneficial
    subplot_analysis = analyze_data_for_subplots(result_df)
    
    if subplot_analysis['needs_subplots'] and len(subplot_analysis['plot_combinations']) > 0:
        print("Subplot Analysis", "info", f"Multiple subplots recommended: {subplot_analysis['subplot_reasons']}")
        
        # Generate subplot code
        subplot_code = generate_multiple_subplot_code(client, user_query, result_df, subplot_analysis, column_mapping)
        
        if subplot_code:
            return subplot_code, True  # Return code and subplot flag
    
    # Fallback to single plot
    print("Plot Generation", "info", "Generating single plot")
    
    df_info = f"""
    DataFrame shape: {result_df.shape}
    Columns: {result_df.columns.tolist()}
    Data types: {result_df.dtypes.to_dict()}
    
    Column Details:
    {json.dumps([{'name': col, 'type': str(result_df[col].dtype), 'sample_values': result_df[col].dropna().head(3).tolist()} for col in result_df.columns], indent=2, default=str)}
    
    First few rows:
    {result_df.head().to_string()}
    """
    
    prompt = f"""
    Generate Python code using Plotly to visualize this data based on the user query:
    
    User Query: "{user_query}"
    
    Data Information:
    {df_info}
    
    Requirements:
    1. Use plotly.express or plotly.graph_objects
    2. The DataFrame variable name should be 'result_df'
    3. Use the EXACT column names as shown in the Columns list above
    4. Return only the Python code to create the figure between codestart and codeend markers
    5. Assign the figure to a variable called 'fig'
    6. Don't include fig.show() or any display commands
    7. Choose the most appropriate chart type for the data and query
    8. Include proper titles and labels
    9. Handle any potential data type issues
    
    Example format:
    codestart
    import plotly.express as px
    fig = px.bar(result_df, x='column1', y='column2', title='Chart Title')
    codeend
    
    Your response must begin with 'codestart' and end with 'codeend'
    """
    
    response = call_groq_api(client, prompt)
    if response:
        # Extract code between codestart and codeend markers
        match = re.search(r'codestart(.*?)codeend', response, re.DOTALL)
        if match:
            code = match.group(1).strip()
            return code, False  # Return code and subplot flag
        else:
            # Fallback extraction
            code = re.sub(r'```python\n?|```\n?|Python Code:\s*', '', response).strip()
            return code, False
    return None, False

def execute_enhanced_plotly_code(code, result_df, is_subplot=False):
    """Enhanced execution function that handles both single plots and subplots"""
    if not code:
        #st.error("No code provided to execute")
        return print("No code provided to execute")
    
    try:
        # Import required libraries in the execution environment
        import numpy as np
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import pandas as pd
        
        # Create a safe execution environment
        exec_globals = {
            'px': px,
            'go': go,
            'make_subplots': make_subplots,
            'result_df': result_df,
            'pd': pd,
            'np': np
        }
        
        # Execute the code
        exec(code, exec_globals)
        
        # Get the figure
        fig = exec_globals.get('fig')
        
        if fig is None:
            # st.error("No figure was created by the code")
            # st.code(code)
            return print("No figure was created by the code",code)
        
        # Additional layout updates for subplots
        if is_subplot:
            fig.update_layout(
                height=600 if fig.layout.annotations and len(fig.layout.annotations) <= 4 else 800,
                showlegend=True,
                margin=dict(t=80, b=60, l=60, r=60)
            )
        
        return fig
        
    except Exception as e:
        # st.error(f"Error executing Plotly code: {str(e)}")
        # st.error("Code that failed:")
        # st.code(code)
        return print(f"Error executing Plotly code: ",{str(e)})

def generate_plotly_code(client, user_query, result_df, column_mapping):
    """Updated wrapper function for backward compatibility"""
    code, is_subplot = generate_enhanced_plotly_code(client, user_query, result_df, column_mapping)
    return code

def execute_plotly_code(code, result_df):
    """Updated wrapper function for backward compatibility"""
    # Try to detect if it's subplot code
    is_subplot = 'make_subplots' in code if code else False
    return execute_enhanced_plotly_code(code, result_df, is_subplot)


#-----------------------------------------extra-------------------------------------------------------
def get_subplot_prompt_enhancement(subplot_type, result_df):
    """Get specific prompt enhancements based on subplot type"""
    
    enhancements = {
        'time_series_subplots': f"""
        For time series subplots:
        - Create one subplot per numeric column showing trends over time
        - Use consistent time axis across all subplots
        - Include trend lines where appropriate
        - Ensure proper date formatting on x-axis
        """,
        
        'category_subplots': f"""
        For category-based subplots:
        - Create separate subplots for each major category
        - Use consistent scales for easy comparison
        - Include totals or averages where relevant
        - Color-code consistently across subplots
        """,
        
        'metrics_dashboard': f"""
        For metrics dashboard:
        - Show key performance indicators in separate panels
        - Use appropriate chart types for each metric (bar, line, gauge)
        - Include summary statistics
        - Maintain consistent styling across panels
        """
    }
    
    return enhancements.get(subplot_type, "")

# Example of specific subplot generation functions for different scenarios
def generate_time_series_subplots(result_df, datetime_col, numeric_cols):
    """Generate specific code for time series subplots"""
    
    code_template = f"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create subplots
rows = {min(len(numeric_cols), 3)}
cols = {2 if len(numeric_cols) > 3 else 1}
fig = make_subplots(
    rows=rows, cols=cols,
    subplot_titles={[f'{col} Over Time' for col in numeric_cols[:6]]},
    shared_xaxes=True
)

# Add traces for each numeric column
"""
    
    for i, col in enumerate(numeric_cols[:6]):  # Limit to 6 subplots
        row = (i // 2) + 1 if len(numeric_cols) > 3 else i + 1
        col_pos = (i % 2) + 1 if len(numeric_cols) > 3 else 1
        
        code_template += f"""
fig.add_trace(
    go.Scatter(x=result_df['{datetime_col}'], y=result_df['{col}'], 
               mode='lines+markers', name='{col}'),
    row={row}, col={col_pos}
)
"""
    
    code_template += """
fig.update_layout(
    title_text="Time Series Dashboard",
    height=600,
    showlegend=True
)

fig.update_xaxes(title_text="Date")
"""
    
    return code_template

def generate_metrics_dashboard(result_df, metrics):
    """Generate specific code for metrics dashboard"""
    
    code_template = f"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create subplots for metrics dashboard
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles={[f'{metric}' for metric in metrics[:6]]},
    specs=[[{{"type": "xy"}}, {{"type": "xy"}}, {{"type": "xy"}}],
           [{{"type": "xy"}}, {{"type": "xy"}}, {{"type": "xy"}}]]
)

"""
    
    for i, metric in enumerate(metrics[:6]):
        row = (i // 3) + 1
        col = (i % 3) + 1
        
        # Determine chart type based on data
        code_template += f"""
# Add {metric} visualization
fig.add_trace(
    go.Bar(x=list(range(len(result_df))), y=result_df['{metric}'], name='{metric}'),
    row={row}, col={col}
)
"""
    
    code_template += """
fig.update_layout(
    title_text="Metrics Dashboard",
    height=700,
    showlegend=False
)
"""
    
    return code_template