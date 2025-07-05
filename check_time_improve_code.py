import pandas as pd
import streamlit as st
from openai import AzureOpenAI
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os
import json
import re
import hashlib
import time
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import AudioConfig, SpeechRecognizer, SpeechConfig
from concurrent.futures import ThreadPoolExecutor
import statsmodels.api as sm
from streamlit_extras.stylable_container import stylable_container

# Load environment variables
env_path = 'C:/Users/n.sureshbabu/OneDrive - ARADA/Intelligent_chatbot/.env'
load_dotenv(dotenv_path=env_path, override=True, verbose=True)


import azure.cognitiveservices.speech as speechsdk

# Timing utilities
class ExecutionTimer:
    """Class to track and display execution times"""
    def __init__(self):
        self.timings: List[Dict[str, float]] = []
        self.current_stage = ""
        
    def start_stage(self, stage_name: str):
        self.current_stage = stage_name
        self.stage_start = time.time()
        
    def end_stage(self):
        if self.current_stage:
            execution_time = time.time() - self.stage_start
            self.timings.append({
                "stage": self.current_stage,
                "time": execution_time
            })
            self.current_stage = ""
            
    def get_timings(self) -> List[Dict[str, float]]:
        return self.timings
    
    def display_timings(self):
        if not self.timings:
            return ""
            
        total_time = sum(t['time'] for t in self.timings)
        timing_text = ["⏱️ Execution Timings:"]
        timing_text.append(f"Total: {total_time:.2f}s")
        
        for timing in self.timings:
            timing_text.append(f"- {timing['stage']}: {timing['time']:.2f}s ({timing['time']/total_time:.1%})")
        
        return "\n".join(timing_text)

def timeit(func):
    """Decorator to measure execution time of functions"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper

# Page configuration
st.set_page_config(
    page_title="Smart Excel Data Chat", 
    layout="wide",
    page_icon="🧠"
)

@dataclass
class ConversationContext:
    """Class to maintain conversation context and entity tracking"""
    entities: Dict[str, Any] = field(default_factory=dict)
    previous_queries: List[Dict[str, str]] = field(default_factory=list)
    data_summary: str = ""
    last_results: Optional[pd.DataFrame] = None
    last_sql_query: str = ""
    
    def add_entity(self, entity_type: str, entity_value: str, properties: Dict[str, Any]):
        """Track an entity mentioned in conversation"""
        if entity_type not in self.entities:
            self.entities[entity_type] = []
        
        existing_entity = next((e for e in self.entities[entity_type] 
                             if e['value'].lower() == entity_value.lower()), None)
        
        if existing_entity:
            existing_entity['properties'].update(properties)
            existing_entity['last_mentioned'] = datetime.now().isoformat()
        else:
            self.entities[entity_type].append({
                'value': entity_value,
                'properties': properties,
                'first_mentioned': datetime.now().isoformat(),
                'last_mentioned': datetime.now().isoformat()
            })
    
    def get_entity_context(self, entity_type: Optional[str] = None) -> str:
        """Get context string for entities"""
        if not self.entities:
            return ""
            
        context_parts = ["\n\nRemembered entities:"]
        
        if entity_type:
            entities = self.entities.get(entity_type, [])
            if entities:
                context_parts.append(f"{entity_type.title()}:")
                for entity in entities:
                    context_parts.append(f"- {entity['value']} (last mentioned: {entity['last_mentioned']})")
        else:
            for ent_type, entities in self.entities.items():
                context_parts.append(f"{ent_type.title()}:")
                for entity in entities[:3]:
                    context_parts.append(f"- {entity['value']}")
                if len(entities) > 3:
                    context_parts.append(f"- ...and {len(entities)-3} more")
        
        return "\n".join(context_parts)
    
    def add_query(self, question: str, answer: str, sql_query: str = ""):
        """Store previous query and answer"""
        self.previous_queries.append({
            'question': question,
            'answer': answer,
            'sql_query': sql_query,
            'timestamp': datetime.now().isoformat()
        })
        if len(self.previous_queries) > 3:
            self.previous_queries.pop(0)
    
    def get_query_context(self) -> str:
        """Get context from previous queries"""
        if not self.previous_queries:
            return ""
            
        context = ["\n\nPrevious questions and answers:"]
        for i, qa in enumerate(self.previous_queries, 1):
            context.append(f"{i}. Q: {qa['question']}")
            context.append(f"   A: {qa['answer'][:200]}...")
            if qa['sql_query']:
                context.append(f"   SQL: {qa['sql_query']}")
        
        return "\n".join(context)


# def analyze_data_for_multiple_plots(result_df):
#     """Analyze DataFrame to determine if multiple plots would be better"""
#     numeric_cols = result_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
#     categorical_cols = result_df.select_dtypes(include=['object', 'category']).columns.tolist()
#     datetime_cols = result_df.select_dtypes(include=['datetime64']).columns.tolist()
    
#     # Check for different axis types/scales
#     multiple_plot_indicators = {
#         'different_scales': False,
#         'mixed_data_types': False,
#         'too_many_categories': False,
#         'suggested_plots': []
#     }
    
#     # Check if numeric columns have very different scales
#     if len(numeric_cols) >= 2:
#         scales = []
#         for col in numeric_cols:
#             col_range = result_df[col].max() - result_df[col].min()
#             scales.append(col_range)
        
#         # If scales differ by more than 2 orders of magnitude
#         max_scale = max(scales)
#         min_scale = min([s for s in scales if s > 0])
#         if max_scale / min_scale > 100:
#             multiple_plot_indicators['different_scales'] = True
    
#     # Check for mixed data types that don't work well together
#     if len(numeric_cols) > 0 and len(categorical_cols) > 0 and len(datetime_cols) > 0:
#         multiple_plot_indicators['mixed_data_types'] = True
    
#     # Check for too many categories
#     for col in categorical_cols:
#         if result_df[col].nunique() > 15:
#             multiple_plot_indicators['too_many_categories'] = True
#             break
    
#     # Suggest plot types based on data
#     if len(numeric_cols) >= 2 and len(categorical_cols) >= 1:
#         multiple_plot_indicators['suggested_plots'].extend(['bar_chart', 'scatter_plot'])
    
#     if len(datetime_cols) > 0 and len(numeric_cols) > 0:
#         multiple_plot_indicators['suggested_plots'].append('time_series')
    
#     if len(categorical_cols) > 0:
#         multiple_plot_indicators['suggested_plots'].append('pie_chart')
    
#     return multiple_plot_indicators

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
    
    system_prompt = f"""
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
    
    response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.1,
            max_tokens=1000
        )
    response =response.choices[0].message.content.strip()
    # Extract code between codestart and codeend markers
    match = re.search(r'codestart(.*?)codeend', response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        return code
    else:
        # Fallback extraction
        code = re.sub(r'```python\n?|```\n?|Python Code:\s*', '', response).strip()
        return code
        

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
    
    system_prompt = f"""
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
    
    response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.1,
            max_tokens=1000
        )
    response =response.choices[0].message.content.strip()
    # Extract code between codestart and codeend markers
    match = re.search(r'codestart(.*?)codeend', response, re.DOTALL)
    if match:
        code = match.group(1).strip()
        return code, False  # Return code and subplot flag
    else:
        # Fallback extraction
        code = re.sub(r'```python\n?|```\n?|Python Code:\s*', '', response).strip()
        return code, False


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
if 'synthesizer' not in st.session_state:
    st.session_state.synthesizer = None
if 'synthesis_future' not in st.session_state:
    st.session_state.synthesis_future = None

def speak_with_azure(text: str):
    """Speak text using Azure TTS, storing the synthesizer in session state"""
    speech_config = speechsdk.SpeechConfig(
        subscription=os.getenv("AZURE_SPEECH_KEY"),
        region=os.getenv("AZURE_SPEECH_REGION")
    )
    speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"
    
    # Store synthesizer and future in session state
    st.session_state.synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    st.session_state.synthesis_future = st.session_state.synthesizer.speak_text_async(text)
    
    # Get the result (this will block until completion or cancellation)
    result = st.session_state.synthesis_future.get()
    
    if result.reason == speechsdk.ResultReason.Canceled:
        st.warning("Speech interrupted")
    elif result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        st.error(f"TTS failed: {result.reason}")

def stop_speaking():
    """Stop any ongoing speech synthesis"""
    if st.session_state.synthesis_future:
        st.session_state.synthesis_future.cancel()
    if st.session_state.synthesizer:
        st.session_state.synthesizer.stop_speaking()
        st.session_state.synthesizer = None
    st.session_state.synthesis_future = None

# In your main code where you display insights:




def transcribe_speech():
    """Safe speech recognition with error handling"""
    try:
        speech_key = os.getenv("AZURE_SPEECH_KEY")
        speech_region = os.getenv("AZURE_SPEECH_REGION")

        if not speech_key or not speech_region:
            st.error("❌ Missing Azure Speech credentials")
            return ""

        speech_config = speechsdk.SpeechConfig(
            subscription=speech_key.strip(),
            region=speech_region.strip()
        )
        
        audio_config = speechsdk.AudioConfig(
            use_default_microphone=True
        )

        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
            language="en-US"
        )

        st.info("🎤 Speak now... (Press Ctrl+C to cancel)")
        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            st.warning("🔇 No speech detected")
        else:
            st.error(f"❌ Recognition failed: {result.reason}")
            
    except Exception as e:
        st.error(f"🚨 Speech recognition error: {str(e)}")
    
    return ""

# Caching functions
@st.cache_data(show_spinner="Loading Excel file...")
def load_excel_file(uploaded_file, sheet_name: str, file_hash: str):
    """Cached function to load Excel file"""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        df = df.dropna(how='all').reset_index(drop=True)
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_resource(show_spinner="Preparing database...")
def create_sqlite_db_cached(df_hash: str, df_columns: List[str], df_dtypes: Dict) -> sqlite3.Connection:
    """Cached function to create SQLite database"""
    return f"db_signature_{df_hash}"

def get_file_hash(uploaded_file) -> str:
    """Generate hash for uploaded file"""
    if uploaded_file is not None:
        uploaded_file.seek(0)
        file_hash = hashlib.md5(uploaded_file.read()).hexdigest()
        uploaded_file.seek(0)
        return file_hash
    return ""

def get_dataframe_hash(df: pd.DataFrame) -> str:
    """Generate hash for dataframe"""
    if df is not None and not df.empty:
        shape_str = f"{df.shape[0]}_{df.shape[1]}"
        cols_str = "_".join(df.columns.astype(str))
        sample_str = str(df.head().values.tobytes()) if len(df) > 0 else ""
        combined = f"{shape_str}_{cols_str}_{sample_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    return ""

def get_credentials_hash(api_key: str, endpoint: str, api_version: str) -> str:
    """Generate hash for credentials"""
    combined = f"{api_key}_{endpoint}_{api_version}"
    return hashlib.md5(combined.encode()).hexdigest()

@st.cache_resource(show_spinner="Initializing Azure OpenAI...")
def initialize_azure_client_cached(api_key: str, endpoint: str, api_version: str, _hash: str) -> Optional[AzureOpenAI]:
    """Cached Azure OpenAI client initialization"""
    try:
        if not api_key or not endpoint:
            return None
            
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        return client
            
    except Exception as e:
        st.error(f"❌ Error initializing Azure OpenAI: {str(e)}")
        return None

# Utility functions
def initialize_azure_client() -> Optional[AzureOpenAI]:
    """Initialize Azure OpenAI client with caching"""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    
    if not api_key or not endpoint:
        with st.sidebar:
            st.subheader("🔑 Azure OpenAI Configuration")
            api_key = st.text_input("API Key:", type="password", value=api_key or "")
            endpoint = st.text_input("Endpoint:", value=endpoint or "")
            
    if not api_key or not endpoint:
        st.warning("⚠️ Please provide Azure OpenAI credentials to continue.")
        return None
    
    creds_hash = get_credentials_hash(api_key, endpoint, api_version)
    return initialize_azure_client_cached(api_key, endpoint, api_version, creds_hash)

def create_sqlite_db(df: pd.DataFrame) -> sqlite3.Connection:
    """Convert DataFrame to in-memory SQLite database"""
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    df_clean = df.copy()
    df_clean.columns = [col.replace(' ', '_').replace('-', '_').replace('.', '_') 
                       for col in df_clean.columns]
    df_clean.to_sql('data', conn, index=False, if_exists='replace')
    return conn

@st.cache_data
def get_table_schema_cached(df_hash: str, columns: List[str]) -> str:
    """Cached table schema generation"""
    schema_parts = ["Table: data"]
    for col_name in columns:
        clean_name = col_name.replace(' ', '_').replace('-', '_').replace('.', '_')
        schema_parts.append(f"- {clean_name} (TEXT)")
    return "\n".join(schema_parts)

def get_table_schema(conn: sqlite3.Connection) -> str:
    """Get table schema information"""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(data)")
    schema_info = cursor.fetchall()
    
    schema_parts = ["Table: data"]
    for col_info in schema_info:
        col_name = col_info[1]
        col_type = col_info[2]
        schema_parts.append(f"- {col_name} ({col_type})")
    return "\n".join(schema_parts)

def is_data_related_query(query: str) -> bool:
    """Check if query is related to data analysis"""
    data_keywords = [
        'show', 'list', 'display', 'get', 'find', 'search', 'look',
        'count', 'sum', 'total', 'average', 'mean', 'median', 'mode',
        'maximum', 'minimum', 'max', 'min', 'std', 'variance',
        'filter', 'where', 'group', 'sort', 'order', 'arrange',
        'top', 'bottom', 'first', 'last', 'highest', 'lowest',
        'compare', 'versus', 'vs', 'between', 'greater', 'less',
        'equal', 'different', 'unique', 'distinct',
        'how many', 'what is', 'which', 'who', 'when', 'where',
        'what are', 'how much', 'tell me about',
        'analysis', 'analyze', 'data', 'rows', 'records', 'values',
        'distribution', 'pattern', 'trend', 'correlation',
        'plot', 'chart', 'graph', 'visualize', 'dashboard',
        'evolution', 'development', 'progress', 'growth', 'change',
        'over time', 'timeline', 'history', 'progression',
        'more about', 'explain', 'elaborate', 'details', 'previous',
        'again', 'follow up', 'follow-up'
    ]
    
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in data_keywords):
        return True
    
    question_patterns = [
        r'\b(what|how|which|who|when|where|why)\b',
        r'\b(can you|could you|please)\b.*\b(show|tell|find|get)\b',
        r'\b(i want to|i need to|help me)\b',
        r'\b(what about|tell me more|what else)\b'
    ]
    
    for pattern in question_patterns:
        if re.search(pattern, query_lower):
            return True
    
    return False

def extract_entities(query: str, result_df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Extract potential entities from query and results"""
    entities = {}
    
    if re.search(r'\b(Majestic Woods|other proper nouns)\b', query, re.I):
        entities['location'] = [{'value': 'Majestic Woods', 'source': 'query'}]
    
    if result_df is not None and not result_df.empty:
        categorical_cols = [col for col in result_df.columns 
                          if result_df[col].dtype == 'object' and result_df[col].nunique() < 50]
        
        for col in categorical_cols:
            col_entities = []
            for val in result_df[col].unique()[:5]:
                if pd.notna(val) and isinstance(val, str) and len(val.strip()) > 2:
                    col_entities.append({'value': val, 'source': f'column: {col}'})
            
            if col_entities:
                entities[col] = col_entities
    
    return entities

def should_create_visualization(query: str, result_df: pd.DataFrame) -> bool:
    """Determine if visualization is needed"""
    if result_df is None or result_df.empty or len(result_df) < 2:
        return False
    
    viz_keywords = [
        'plot', 'chart', 'graph', 'visualize', 'visual', 'show trend', 'compare', 
        'distribution', 'over time', 'relationship', 'correlation', 'dashboard',
        'evolution', 'development', 'progress', 'growth', 'change'
    ]
    
    query_lower = query.lower()
    has_viz_keyword = any(keyword in query_lower for keyword in viz_keywords)
    
    has_multiple_rows = len(result_df) > 1
    has_numeric_data = any(pd.api.types.is_numeric_dtype(result_df[col]) for col in result_df.columns)
    has_categorical = any(result_df[col].dtype == 'object' for col in result_df.columns)
    
    auto_viz_conditions = [
        has_viz_keyword,
        (has_multiple_rows and has_numeric_data and len(result_df) <= 100),
        (has_categorical and has_numeric_data and len(result_df) <= 50),
        ('top' in query_lower or 'bottom' in query_lower) and has_numeric_data,
        ('compare' in query_lower or 'vs' in query_lower) and has_multiple_rows
    ]
    
    return any(auto_viz_conditions)

def generate_sql_query(client: AzureOpenAI, user_query: str, schema: str, 
                      context: ConversationContext) -> Dict[str, Any]:
    """Generate SQL query from natural language"""
    try:
        previous_queries = context.get_query_context()
        entity_context = context.get_entity_context()
        sql_dialect = "SQLite"

        system_prompt = f"""You are an expert SQL analyst. Convert natural language to SQL queries.

Database Schema: {schema}
Context: {previous_queries} {entity_context}
SQL Dialect: {sql_dialect}

RULES:
1. Only SELECT statements - no WITH, CTEs, window functions, or UNION
2. Table name: 'data'
3. Column names: case-sensitive, wrap special chars/spaces in double quotes
4. Handle NULL values with IS NOT NULL or COALESCE
5. Use LIMIT 100 unless user asks for all
6. Return only SQL query, no explanations
7. Use GROUP BY for all non-aggregate columns in SELECT
8. Use HAVING for aggregate conditions, WHERE for row filtering
9. Ensure ORDER BY columns exist in SELECT clause
10. Use subqueries for comparisons instead of window functions
11. For time series: use strftime('%Y-%m', date_column) for SQLite
12. For conditional counts: use CASE WHEN within SUM/COUNT
13. For top/bottom queries: use ORDER BY with LIMIT
14. For ranking: use correlated subqueries instead of ROW_NUMBER()
15. Always filter out NULL values before aggregations
16. Use CAST(... AS REAL) for percentage calculations
17. Handle follow-up questions by referencing previous context
18. Never use multiple statements separated by semicolons
19. Avoid reserved keywords as column names unless double-quoted
20. Use single SELECT statement only - no compound queries"""
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        sql_query = response.choices[0].message.content.strip()
        sql_query = re.sub(r'^```sql\s*', '', sql_query, flags=re.IGNORECASE)
        sql_query = re.sub(r'\s*```$', '', sql_query)
        sql_query = sql_query.strip()
        
        return {
            "success": True,
            "sql_query": sql_query
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def execute_sql_query(conn: sqlite3.Connection, sql_query: str) -> Dict[str, Any]:
    """Execute SQL query safely"""
    try:
        sql_lower = sql_query.lower().strip()
        
        blocked_keywords = [
            'insert', 'update', 'delete', 'drop', 'alter', 'create', 
            'truncate', 'replace', 'grant', 'revoke'
        ]
        
        if not sql_lower.startswith('select'):
            return {
                "success": False,
                "error": "Only SELECT queries are allowed"
            }
            
        if any(f' {kw} ' in f' {sql_lower} ' for kw in blocked_keywords):
            return {
                "success": False,
                "error": "Query contains blocked SQL keywords"
            }
        
        result_df = pd.read_sql_query(sql_query, conn)
        
        return {
            "success": True,
            "data": result_df,
            "row_count": len(result_df),
            "sql_query": sql_query
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def generate_insights(client: AzureOpenAI, user_query: str, result_df: pd.DataFrame, 
                     context: ConversationContext) -> str:
    """Generate insights from query results"""
    try:
        if result_df.empty:
            return "No data found matching your query. Try rephrasing your question or check if the data contains what you're looking for."
        
        data_summary = f"Query returned {len(result_df)} rows with columns: {list(result_df.columns)}"
        
        numeric_cols = result_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats_summary = "\nNumeric data summary:"
            for col in numeric_cols:
                stats_summary += f"\n- {col}: min={result_df[col].min():.2f}, max={result_df[col].max():.2f}, avg={result_df[col].mean():.2f}"
        else:
            stats_summary = ""
        
        if len(result_df) <= 10:
            data_summary += f"\nComplete data:\n{result_df.to_string(index=False)}"
        else:
            data_summary += f"\nFirst 5 rows:\n{result_df.head().to_string(index=False)}"
            if len(result_df) > 5:
                data_summary += f"\nLast 5 rows:\n{result_df.tail().to_string(index=False)}"
        
        data_summary += stats_summary
        
        previous_queries = context.get_query_context()
        entity_context = context.get_entity_context()
        
        system_prompt = f"""You are a data analyst. Interpret the result concisely and clearly.

User Question:
{user_query}

Context:
{previous_queries}
{entity_context}

Data Summary:
{data_summary}

Instructions:

Begin with a direct answer to the user's question.

Mention only the most relevant numbers, trends, or anomalies.

Refer to previous context only if it strengthens the insight.

Avoid speculation or details not present in the data.

Write as one clear paragraph. No bullet points, no headings. 
make sure insights are not telling complete number insted just address as million or billion or other way also it should only talk about project not the sql or other things

"""

        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_MODEL", "gpt-4.1"),
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Based on the data analysis of {len(result_df) if not result_df.empty else 0} records, here are the key findings from your query."
def should_visualize(query, result_df):
    """Determine if visualization would be helpful for this query and data"""
    # Check if the query suggests visualization would be helpful
    visualization_triggers = [
        'show', 'display', 'visualize', 'plot', 'chart', 
        'graph', 'trend', 'compare', 'distribution',
        'over time', 'by category', 'relationship'
    ]
    
    query_lower = query.lower()
    if any(trigger in query_lower for trigger in visualization_triggers):
        return True
    
    # Check if the data is suitable for visualization
    if len(result_df) < 2:
        return False  # Not enough data points
    
    numeric_cols = result_df.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        return False  # No numeric columns to plot
    
    return True
def create_enhanced_visualization(result_df: pd.DataFrame, query: str, 
                                client: Optional[AzureOpenAI] = None, 
                                context: Optional[ConversationContext] = None) -> Optional[go.Figure]:
    """Create visualization with LLM suggestions"""
    try:
        if result_df.empty or len(result_df) < 2:
            return None
            
        numeric_cols = [col for col in result_df.columns if pd.api.types.is_numeric_dtype(result_df[col])]
        categorical_cols = [col for col in result_df.columns if result_df[col].dtype in ['object', 'category'] and result_df[col].nunique() < 50]
        
        date_cols = []
        for col in result_df.columns:
            if (pd.api.types.is_datetime64_any_dtype(result_df[col]) or 
                any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day']) or
                (result_df[col].dtype == 'object' and 
                 any(str(val).count('/') == 2 or str(val).count('-') == 2 for val in result_df[col].dropna().head()))):
                date_cols.append(col)
        
        query_lower = query.lower()
        
        # Evolution/Development charts
        if (any(keyword in query_lower for keyword in ['evolution', 'development', 'progress', 'over time', 'timeline', 'trend']) 
            and (date_cols or categorical_cols) and numeric_cols):
            
            x_col = date_cols[0] if date_cols else categorical_cols[0]
            y_col = numeric_cols[0]
            
            if date_cols:
                if not pd.api.types.is_datetime64_any_dtype(result_df[x_col]):
                    try:
                        result_df[x_col] = pd.to_datetime(result_df[x_col])
                    except:
                        pass
                
                fig = px.line(result_df, x=x_col, y=y_col, 
                             title=f"Evolution of {y_col} over time",
                             markers=True)
                fig.update_layout(
                    xaxis_title=x_col.replace('_', ' ').title(),
                    yaxis_title=y_col.replace('_', ' ').title(),
                    hovermode='x unified',
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1m", step="month", stepmode="backward"),
                                dict(count=6, label="6m", step="month", stepmode="backward"),
                                dict(count=1, label="YTD", step="year", stepmode="todate"),
                                dict(count=1, label="1y", step="year", stepmode="backward"),
                                dict(step="all")
                            ])
                        ),
                        rangeslider=dict(visible=True),
                        type="date"
                    )
                )
                return fig
            else:
                fig = px.line(result_df, x=x_col, y=y_col, 
                             title=f"Development of {y_col} by {x_col}",
                             markers=True)
                fig.update_xaxes(tickangle=45)
                return fig
        
        # Enhanced bar charts
        elif categorical_cols and numeric_cols and len(result_df) <= 30:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            if 'top' in query_lower or 'highest' in query_lower:
                result_df = result_df.nlargest(min(20, len(result_df)), num_col)
                title = f"Top {len(result_df)} {cat_col} by {num_col}"
            elif 'bottom' in query_lower or 'lowest' in query_lower:
                result_df = result_df.nsmallest(min(20, len(result_df)), num_col)
                title = f"Bottom {len(result_df)} {cat_col} by {num_col}"
            else:
                title = f"{num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}"
            
            color_col = None
            if len(categorical_cols) > 1:
                color_col = categorical_cols[1]
            
            fig = px.bar(result_df, x=cat_col, y=num_col, color=color_col,
                        title=title,
                        text=num_col)
            fig.update_xaxes(tickangle=45)
            fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            return fig
        
        # Enhanced pie/donut charts
        elif len(categorical_cols) == 1 and len(result_df) <= 15 and 'distribution' in query_lower:
            cat_col = categorical_cols[0]
            if len(numeric_cols) > 0:
                fig = px.pie(result_df, names=cat_col, values=numeric_cols[0],
                           title=f"Distribution of {numeric_cols[0]} by {cat_col}",
                           hole=0.3)
            else:
                value_counts = result_df[cat_col].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f"Distribution of {cat_col}",
                           hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            return fig
        
        # Enhanced scatter plots
        elif (len(numeric_cols) >= 2 and len(result_df) <= 200 and 
              any(keyword in query_lower for keyword in ['relationship', 'correlation', 'vs', 'versus', 'against'])):
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            
            color_col = None
            if categorical_cols:
                color_col = categorical_cols[0]
            
            fig = px.scatter(result_df, x=x_col, y=y_col, color=color_col,
                            title=f"Relationship between {x_col} and {y_col}",
                            trendline="lowess" if len(result_df) > 10 else None)
            
            if len(result_df) > 2:
                corr = result_df[[x_col, y_col]].corr().iloc[0,1]
                fig.add_annotation(
                    x=0.95, y=0.95,
                    xref="paper", yref="paper",
                    text=f"Correlation: {corr:.2f}",
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )
            return fig
        
        # Enhanced box plots
        elif (categorical_cols and numeric_cols and 
              any(keyword in query_lower for keyword in ['distribution', 'spread', 'range', 'variance'])):
            fig = px.box(result_df, x=categorical_cols[0], y=numeric_cols[0],
                        title=f"Distribution of {numeric_cols[0]} by {categorical_cols[0]}",
                        points="all" if len(result_df) < 50 else False)
            fig.update_xaxes(tickangle=45)
            return fig
        
        # Enhanced histogram
        elif len(numeric_cols) >= 1 and len(result_df) > 10:
            fig = px.histogram(result_df, x=numeric_cols[0], 
                             title=f"Distribution of {numeric_cols[0]}",
                             nbins=min(30, max(10, int(len(result_df)/5))),
                             marginal="rug",
                             hover_data=result_df.columns)
            return fig
            
        # Default to showing first two numeric columns
        elif len(numeric_cols) >= 2:
            fig = px.scatter(result_df, x=numeric_cols[0], y=numeric_cols[1],
                           title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
            return fig
        
        return None
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def process_user_query(prompt: str, client: AzureOpenAI, conn: sqlite3.Connection, 
                      context: ConversationContext) -> Tuple[str, pd.DataFrame, Optional[go.Figure]]:
    """Process user query with timing measurements"""
    timer = st.session_state.execution_timer
    timer.timings = []  # Reset timings for new query
    
    try:
        # Generate SQL query
        timer.start_stage("SQL Generation")
        schema = get_table_schema(conn)
        sql_result = generate_sql_query(client, prompt, schema, context)
        timer.end_stage()
        
        if not sql_result["success"]:
            raise Exception(f"SQL generation failed: {sql_result['error']}")
        
        # Execute SQL query
        timer.start_stage("SQL Execution")
        query_result = execute_sql_query(conn, sql_result["sql_query"])
        timer.end_stage()
        
        if not query_result["success"]:
            raise Exception(f"Query execution failed: {query_result['error']}")
        
        result_df = query_result["data"]
        sql_query = query_result.get("sql_query", "")
        
        # Extract entities
        timer.start_stage("Entity Extraction")
        entities = extract_entities(prompt, result_df)
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                context.add_entity(
                    entity_type=entity_type,
                    entity_value=entity['value'],
                    properties={'source': entity['source']}
                )
        timer.end_stage()
        
        # Generate insights
        timer.start_stage("Insight Generation")
        insights = generate_insights(client, prompt, result_df, context)
        timer.end_stage()
        
        # Create visualization
        fig = None
        timer.start_stage("Visualization Creation")
        if not result_df.empty:
            #with timer.measure("visualization_generation"):
                # Check if visualization would be helpful
            if should_visualize(prompt, result_df):
                # Generate column mapping if needed
                column_mapping = {col: col for col in result_df.columns}
                
                # Generate Plotly code
                plotly_code, is_subplot = generate_enhanced_plotly_code(
                    client, prompt, result_df, column_mapping
                )
                
                if plotly_code:
                    # Execute the Plotly code
                    fig = execute_enhanced_plotly_code(plotly_code, result_df, is_subplot)
        timer.end_stage()
        # Update context
        timer.start_stage("Context Update")
        context.add_query(
            question=prompt,
            answer=insights,
            sql_query=sql_query
        )
        context.last_results = result_df
        context.last_sql_query = sql_query
        timer.end_stage()
        
        return insights, result_df, fig
        
    except Exception as e:
        raise e


# def main():
#     # Add this at the beginning of your main function
#     st.markdown("""
#     <style>
#         /* Main container styling */
#         .stApp {
#             background-color: #f5f7fa;
#         }
        
#         /* Sidebar styling */
#         [data-testid="stSidebar"] {
#             background-color: #2c3e50 !important;
#             color: white !important;
#         }
        
#         [data-testid="stSidebar"] .stMarkdown h1,
#         [data-testid="stSidebar"] .stMarkdown h2,
#         [data-testid="stSidebar"] .stMarkdown h3 {
#             color: white !important;
#         }
        
#         /* Chat message styling */
#         [data-testid="stChatMessage"] {
#             padding: 12px 16px;
#             border-radius: 12px;
#             margin-bottom: 8px;
#             max-width: 80%;
#         }
        
#         [data-testid="stChatMessage"] p {
#             margin: 0;
#         }
        
#         .stChatMessage.user {
#             background-color: #e3f2fd;
#             margin-left: auto;
#             border-bottom-right-radius: 4px;
#         }
        
#         .stChatMessage.assistant {
#             background-color: #f1f1f1;
#             margin-right: auto;
#             border-bottom-left-radius: 4px;
#         }
        
#         /* Button styling */
#         .stButton button {
#             border-radius: 8px !important;
#             transition: all 0.3s ease;
#         }
        
#         .stButton button:hover {
#             transform: translateY(-2px);
#             box-shadow: 0 4px 8px rgba(0,0,0,0.1);
#         }
        
#         /* Input field styling */
#         .stTextInput input {
#             border-radius: 12px !important;
#             padding: 10px 16px !important;
#         }
        
#         /* Dataframe styling */
#         .stDataFrame {
#             border-radius: 8px;
#             box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         }
        
#         /* Custom scrollbar */
#         ::-webkit-scrollbar {
#             width: 8px;
#         }
        
#         ::-webkit-scrollbar-track {
#             background: #f1f1f1;
#         }
        
#         ::-webkit-scrollbar-thumb {
#             background: #888;
#             border-radius: 4px;
#         }
        
#         ::-webkit-scrollbar-thumb:hover {
#             background: #555;
#         }
        
#         /* Custom animation for loading */
#         @keyframes pulse {
#             0% { opacity: 0.6; }
#             50% { opacity: 1; }
#             100% { opacity: 0.6; }
#         }
        
#         .stSpinner > div {
#             animation: pulse 1.5s infinite ease-in-out;
#         }
        
#         /* Voice button styling */
#         .voice-btn {
#             position: fixed;
#             right: 20px;
#             bottom: 80px;
#             z-index: 100;
#             background: linear-gradient(135deg, #6e8efb, #a777e3);
#             color: white;
#             border-radius: 50%;
#             width: 50px;
#             height: 50px;
#             display: flex;
#             align-items: center;
#             justify-content: center;
#             box-shadow: 0 4px 12px rgba(0,0,0,0.2);
#             cursor: pointer;
#             transition: all 0.3s ease;
#         }
        
#         .voice-btn:hover {
#             transform: scale(1.1);
#         }
        
#         /* Tooltip styling */
#         .tooltip {
#             position: relative;
#             display: inline-block;
#         }
        
#         .tooltip .tooltiptext {
#             visibility: hidden;
#             width: 120px;
#             background-color: #555;
#             color: #fff;
#             text-align: center;
#             border-radius: 6px;
#             padding: 5px;
#             position: absolute;
#             z-index: 1;
#             bottom: 125%;
#             left: 50%;
#             margin-left: -60px;
#             opacity: 0;
#             transition: opacity 0.3s;
#         }
        
#         .tooltip:hover .tooltiptext {
#             visibility: visible;
#             opacity: 1;
#         }
#     </style>
    
#     <script>
#     // Add custom JavaScript for enhanced interactivity
#     document.addEventListener('DOMContentLoaded', function() {
#         // Add pulse animation to voice button when listening
#         const voiceBtn = document.querySelector('[title="Voice Input"]');
#         if (voiceBtn) {
#             voiceBtn.addEventListener('click', function() {
#                 this.style.animation = 'pulse 0.8s infinite';
#                 setTimeout(() => {
#                     this.style.animation = '';
#                 }, 3000);
#             });
#         }
        
#         // Smooth scroll to bottom of chat
#         function scrollToBottom() {
#             const chatContainer = document.querySelector('[data-testid="stChatMessageContainer"]');
#             if (chatContainer) {
#                 chatContainer.scrollTop = chatContainer.scrollHeight;
#             }
#         }
        
#         // Scroll to bottom when new message arrives
#         const observer = new MutationObserver(scrollToBottom);
#         const config = { childList: true, subtree: true };
#         const target = document.querySelector('[data-testid="stChatMessageContainer"]');
#         if (target) {
#             observer.observe(target, config);
#         }
        
#         // Add custom hover effects to buttons
#         const buttons = document.querySelectorAll('.stButton button, button[role="button"]');
#         buttons.forEach(btn => {
#             btn.addEventListener('mouseenter', () => {
#                 btn.style.transform = 'translateY(-2px)';
#                 btn.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
#             });
#             btn.addEventListener('mouseleave', () => {
#                 btn.style.transform = '';
#                 btn.style.boxShadow = '';
#             });
#         });
        
#         // Add custom class to chat messages based on role
#         const chatMessages = document.querySelectorAll('[data-testid="stChatMessage"]');
#         chatMessages.forEach(msg => {
#             const role = msg.querySelector('[role="img"]')?.getAttribute('aria-label');
#             if (role === 'user') {
#                 msg.classList.add('user');
#             } else if (role === 'assistant') {
#                 msg.classList.add('assistant');
#             }
#         });
#     });
#     </script>
#     """, unsafe_allow_html=True)

#     # Rest of your existing main() function code...
#     """Main application with timing measurements"""
#     st.title("🧠 Smart Excel Data Chat")
#     st.markdown("Upload Excel data and ask intelligent questions - now with performance metrics!")
#     if "tts_intro_done" not in st.session_state:
#         speak_command = (
#             "Hi, I’m your in-house sales analyzer. Please upload your Excel file to begin."
#         )
#         speak_with_azure(speak_command)
#         st.session_state["tts_intro_done"] = True
#     # Initialize session state
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#     if 'context' not in st.session_state:
#         st.session_state.context = ConversationContext()
#     if 'execution_timer' not in st.session_state:
#         st.session_state.execution_timer = ExecutionTimer()
#     if 'current_file_hash' not in st.session_state:
#         st.session_state.current_file_hash = ""
#     if 'current_df_hash' not in st.session_state:
#         st.session_state.current_df_hash = ""
#     if 'cached_df' not in st.session_state:
#         st.session_state.cached_df = None
#     if 'cached_conn' not in st.session_state:
#         st.session_state.cached_conn = None
#     if 'tts_query_ready_done' not in st.session_state:
#         st.session_state.tts_query_ready_done = False
    
#     # Sidebar
#     with st.sidebar:
#         st.header("📂 Upload Data")
#         uploaded_file = st.file_uploader(
#             "Choose Excel file",
#             type=['xlsx', 'xls'],
#             help="Upload your Excel file to start intelligent data chat"
#         )
        
#         # Data loading with caching
#         if uploaded_file is not None:
#             try:
#                 new_file_hash = get_file_hash(uploaded_file)
                
#                 if new_file_hash != st.session_state.current_file_hash:
#                     st.session_state.current_file_hash = new_file_hash
                    
#                     excel_file = pd.ExcelFile(uploaded_file)
#                     sheet_name = st.selectbox("📋 Select Sheet:", excel_file.sheet_names)
                    
#                     df, error = load_excel_file(uploaded_file, sheet_name, new_file_hash)
                    
#                     if error:
#                         st.error(f"❌ Error loading file: {error}")
#                     elif df is not None and not df.empty:
#                         new_df_hash = get_dataframe_hash(df)
                        
#                         if new_df_hash != st.session_state.current_df_hash:
#                             st.session_state.current_df_hash = new_df_hash
#                             st.session_state.cached_df = df
                            
#                             if hasattr(st.session_state, 'cached_conn') and st.session_state.cached_conn:
#                                 try:
#                                     st.session_state.cached_conn.close()
#                                 except:
#                                     pass
#                             st.session_state.cached_conn = create_sqlite_db(df)
                            
#                             st.session_state.context = ConversationContext()
#                             st.session_state.context.data_summary = f"Data loaded with {len(df)} rows and {len(df.columns)} columns"
#                             st.session_state.messages = []
                            
#                             st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
#                             if not st.session_state.tts_query_ready_done:
#                                 speak_with_azure("Great! Your data is ready. You can now type or speak your query.")
#                                 st.session_state.tts_query_ready_done = True
#                         else:
#                             st.info("📊 Using cached data (no changes detected)")
#                             df = st.session_state.cached_df
#                 else:
#                     if st.session_state.cached_df is not None:
#                         excel_file = pd.ExcelFile(uploaded_file)
#                         sheet_name = st.selectbox("📋 Select Sheet:", excel_file.sheet_names)
#                         df = st.session_state.cached_df
#                         st.info("📊 Using cached data")
#                     else:
#                         df = None
                
#                 if st.session_state.cached_df is not None:
#                     df = st.session_state.cached_df
                    
#                     with st.expander("📊 Data Preview"):
#                         st.dataframe(df.head())
                        
#                     with st.expander("📋 Column Info"):
#                         col_info = []
#                         for col in df.columns:
#                             col_type = str(df[col].dtype)
#                             null_count = df[col].isnull().sum()
#                             unique_count = df[col].nunique()
#                             col_info.append({
#                                 'Column': col,
#                                 'Type': col_type,
#                                 'Null Values': null_count,
#                                 'Unique Values': unique_count
#                             })
#                         st.dataframe(pd.DataFrame(col_info))
                        
#                     with st.expander("📈 Data Statistics"):
#                         st.write(f"**Total Rows:** {len(df):,}")
#                         st.write(f"**Total Columns:** {len(df.columns)}")
#                         st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                        
#                         numeric_cols = df.select_dtypes(include=['number']).columns
#                         categorical_cols = df.select_dtypes(include=['object']).columns
                        
#                         if len(numeric_cols) > 0:
#                             st.write(f"**Numeric Columns:** {len(numeric_cols)}")
#                         if len(categorical_cols) > 0:
#                             st.write(f"**Categorical Columns:** {len(categorical_cols)}")
                        
#             except Exception as e:
#                 st.error(f"❌ Error processing file: {str(e)}")
        
#         # Clear buttons
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("🔄 Clear Chat"):
#                 st.session_state.messages = []
#                 st.session_state.context = ConversationContext()
#                 for key in list(st.session_state.keys()):
#                     if key.startswith("show_viz_"):
#                         del st.session_state[key]
#                 st.rerun()
        
#         with col2:
#             if st.button("🗑️ Clear Cache"):
#                 st.session_state.current_file_hash = ""
#                 st.session_state.current_df_hash = ""
#                 st.session_state.cached_df = None
#                 if st.session_state.cached_conn:
#                     st.session_state.cached_conn.close()
#                 st.session_state.cached_conn = None
#                 st.session_state.messages = []
#                 st.session_state.context = ConversationContext()
#                 st.cache_data.clear()
#                 st.cache_resource.clear()
#                 st.success("🧹 Cache cleared!")
#                 st.rerun()
        
#         # AI Configuration
#         st.header("🤖 AI Setup")
#         client = initialize_azure_client()
        
#         # Performance Info
#         if st.session_state.cached_df is not None:
#             st.header("⚡ Performance")
#             st.success("✅ Data cached in memory")
#             st.success("✅ Database connection ready")
#             if client:
#                 st.success("✅ AI client initialized")
        
#         # Example queries
#         st.header("💡 Example Queries")
#         st.markdown("""
#         - Show me the top 10 records by sales
#         - What's the average price by category?
#         """)
    
#     # Main interface
#     if st.session_state.cached_df is None:
#         st.info("👆 Please upload an Excel file to start chatting with your data!")
#         return

#     if client is None:
#         st.error("❌ Please configure Azure OpenAI credentials in the sidebar")
#         return

#     # Display chat history
#     for i, message in enumerate(st.session_state.messages):
#         with st.chat_message(message["role"]):
            
#             if message["role"] == "assistant":
#                 col1, col2 = st.columns([0.95, 0.05])
                
#                 with col1:
#                     st.write(message["content"])
                    
                
#                 if "figure" in message and message["figure"] is not None:
#                     with col2:
#                         viz_key = f"viz_{i}"
#                         if st.button("📊", key=viz_key, help="Click to view chart"):
#                             st.session_state[f"show_viz_{i}"] = not st.session_state.get(f"show_viz_{i}", False)
                    
#                     if st.session_state.get(f"show_viz_{i}", False):
#                         st.plotly_chart(message["figure"], use_container_width=True)
                
#                 if "dataframe" in message and not message["dataframe"].empty:
#                     if len(message["dataframe"]) <= 20:
#                         st.dataframe(message["dataframe"], use_container_width=True)
#                     else:
#                         st.write(f"*Showing first 10 of {len(message['dataframe'])} results:*")
#                         st.dataframe(message["dataframe"].head(10), use_container_width=True)
#             else:
#                 st.write(message["content"])
     
#     # Chat input
#     if st.button("🎙️", help="Voice Input"):
#         with st.spinner("Listening..."):
#             voice_text = transcribe_speech()
#             if voice_text:
#                 st.session_state.voice_prompt = voice_text
    
#     user_input = st.chat_input("Ask about your data... (e.g., 'Show sales trends')")
    
#     prompt = st.session_state.pop("voice_prompt", None) if "voice_prompt" in st.session_state else user_input

#     if prompt:
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         with st.chat_message("user"):
#             st.write(prompt)
        
#         with st.chat_message("assistant"):
#             with st.spinner("🤔 Analyzing your data..."):
#                 if not is_data_related_query(prompt):
#                     response = """I'm your data analysis assistant! I can help you explore and understand your data. 

# Try asking questions like:
# - "Show me the top 10 records by [column name]"
# - "What's the average/sum/count of [column]?"
# - "Evolution of [column] over time"
# - "Compare [column] by [category]"
# - "Find records where [condition]"
# - "Tell me more about [entity from previous results]"
# - "Explain the previous results in more detail"

# What would you like to know about your data?"""

#                     st.write(response)
#                     st.session_state.messages.append({"role": "assistant", "content": response})
#                     return
                
#                 try:
#                     if not st.session_state.cached_conn:
#                         st.error("❌ Database connection not available. Please reload your data.")
#                         return
                    
#                     insights, result_df, fig = process_user_query(
#                         prompt, 
#                         client, 
#                         st.session_state.cached_conn, 
#                         st.session_state.context
#                     )
                    
#                     with st.expander("⏱️ Performance Metrics"):
#                         st.code(st.session_state.execution_timer.display_timings())
                
#                     col1, col2 = st.columns([0.95, 0.05])
                    
#                     with col1:
#                         if fig:
#                             st.plotly_chart(fig, use_container_width=True)
#                         st.write(insights)
#                         speak_with_azure(insights)
                    
#                     # if fig:
#                     #     message_idx = len(st.session_state.messages)
#                     #     viz_key = f"viz_{message_idx}"
#                     #     if st.button("📊", key=viz_key, help="Click to view visualization"):
#                     #         st.session_state[f"show_viz_{message_idx}"] = not st.session_state.get(f"show_viz_{message_idx}", False)
                    
#                     # # Display the visualization if toggled on
#                     # if st.session_state.get(f"show_viz_{message_idx}", False):
                        
                    
#                     if not result_df.empty:
#                         if len(result_df) <= 20:
#                             st.dataframe(result_df, use_container_width=True)
#                         else:
#                             st.write(f"*Showing first 10 of {len(result_df)} results:*")
#                             st.dataframe(result_df.head(10), use_container_width=True)
                    
#                     message_data = {
#                         "role": "assistant", 
#                         "content": insights,
#                         "dataframe": result_df if len(result_df) <= 50 else result_df.head(20)
#                     }
#                     if fig:
#                         message_data["figure"] = fig
                        
#                     st.session_state.messages.append(message_data)
                    
#                 except Exception as e:
#                     error_msg = f"⚠️ I encountered an issue processing your request: {str(e)}"
#                     st.error(error_msg)
#                     st.session_state.messages.append({"role": "assistant", "content": error_msg})

# if __name__ == "__main__":
#     main()
# def main():
#     # Custom CSS and JavaScript for enhanced UI
#     st.markdown("""
#     <style>
#         /* Main container styling */
#         .stApp {
#             background-color: #f5f7fa;
#         }
        
#         /* Sidebar styling */
#         [data-testid="stSidebar"] {
#             background-color: #2c3e50 !important;
#             color: white !important;
#         }
        
#         [data-testid="stSidebar"] .stMarkdown h1,
#         [data-testid="stSidebar"] .stMarkdown h2,
#         [data-testid="stSidebar"] .stMarkdown h3 {
#             color: white !important;
#         }
        
#         /* Chat message styling */
#         [data-testid="stChatMessage"] {
#             padding: 12px 16px;
#             border-radius: 12px;
#             margin-bottom: 8px;
#             max-width: 80%;
#         }
        
#         [data-testid="stChatMessage"] p {
#             margin: 0;
#         }
        
#         .stChatMessage.user {
#             background-color: #e3f2fd;
#             margin-left: auto;
#             border-bottom-right-radius: 4px;
#         }
        
#         .stChatMessage.assistant {
#             background-color: #f1f1f1;
#             margin-right: auto;
#             border-bottom-left-radius: 4px;
#         }
        
#         /* Button styling */
#         .stButton button {
#             border-radius: 8px !important;
#             transition: all 0.3s ease;
#         }
        
#         .stButton button:hover {
#             transform: translateY(-2px);
#             box-shadow: 0 4px 8px rgba(0,0,0,0.1);
#         }
        
#         /* Input field styling */
#         .stTextInput input {
#             border-radius: 12px !important;
#             padding: 10px 16px !important;
#         }
        
#         /* Dataframe styling */
#         .stDataFrame {
#             border-radius: 8px;
#             box-shadow: 0 2px 4px rgba(0,0,0,0.1);
#         }
        
#         /* Custom scrollbar */
#         ::-webkit-scrollbar {
#             width: 8px;
#         }
        
#         ::-webkit-scrollbar-track {
#             background: #f1f1f1;
#         }
        
#         ::-webkit-scrollbar-thumb {
#             background: #888;
#             border-radius: 4px;
#         }
        
#         ::-webkit-scrollbar-thumb:hover {
#             background: #555;
#         }
        
#         /* Custom animation for loading */
#         @keyframes pulse {
#             0% { opacity: 0.6; }
#             50% { opacity: 1; }
#             100% { opacity: 0.6; }
#         }
        
#         .stSpinner > div {
#             animation: pulse 1.5s infinite ease-in-out;
#         }
        
#         /* Voice-first input container - initially centered */
#         .voice-first-input-container {
#             position: fixed;
#             bottom: 20px;
#             left: 50%;
#             transform: translateX(-50%);
#             z-index: 1000;
#             display: flex;
#             flex-direction: column;
#             align-items: center;
#             gap: 10px;
#             background: rgba(255, 255, 255, 0.95);
#             backdrop-filter: blur(10px);
#             padding: 15px 25px;
#             border-radius: 20px;
#             box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
#             border: 1px solid rgba(255, 255, 255, 0.2);
#             transition: all 0.3s ease;
#         }
        
#         /* Voice input container - moved to right corner after messages */
#         .voice-input-container-right {
#             position: fixed;
#             bottom: 20px;
#             right: 20px;
#             left: auto;
#             transform: none;
#             z-index: 1000;
#             display: flex;
#             flex-direction: column;
#             align-items: center;
#             gap: 10px;
#             background: rgba(255, 255, 255, 0.95);
#             backdrop-filter: blur(10px);
#             padding: 15px 25px;
#             border-radius: 20px;
#             box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
#             border: 1px solid rgba(255, 255, 255, 0.2);
#             transition: all 0.3s ease;
#         }
        
#         /* Ask about your data text */
#         .ask-data-text {
#             font-size: 14px;
#             color: #666;
#             text-align: center;
#             margin: 0;
#             font-weight: 500;
#         }
        
#         /* Main microphone button - large and prominent */
#         .main-mic-button {
#             width: 60px !important;
#             height: 60px !important;
#             border-radius: 50% !important;
#             background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
#             border: none !important;
#             color: white !important;
#             font-size: 24px !important;
#             display: flex !important;
#             align-items: center !important;
#             justify-content: center !important;
#             cursor: pointer !important;
#             transition: all 0.3s ease !important;
#             box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
#             overflow: hidden !important;
#         }
        
#         .main-mic-button:hover {
#             transform: scale(1.1) !important;
#             box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
#         }
        
#         .main-mic-button:active {
#             transform: scale(0.95) !important;
#         }
        
#         /* Microphone image styling */
#         .mic-image {
#             width: 40px;
#             height: 40px;
#             object-fit: contain;
#             filter: brightness(0) invert(1);
#         }
        
#         /* Text input overlay when keyboard mode is active */
#         .text-input-overlay {
#             position: fixed;
#             bottom: 0;
#             left: 0;
#             right: 0;
#             background: rgba(255, 255, 255, 0.98);
#             backdrop-filter: blur(10px);
#             padding: 20px;
#             box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.1);
#             z-index: 1001;
#             border-top: 1px solid rgba(0, 0, 0, 0.1);
#         }
        
#         .text-input-overlay .stTextInput {
#             margin-bottom: 0;
#         }
        
#         .text-input-overlay .stTextInput input {
#             border: 2px solid #667eea !important;
#             padding: 12px 16px !important;
#             font-size: 16px !important;
#         }
        
#         /* Close button for text input overlay */
#         .close-text-input {
#             position: absolute;
#             top: 10px;
#             right: 15px;
#             background: none !important;
#             border: none !important;
#             font-size: 20px !important;
#             color: #666 !important;
#             cursor: pointer !important;
#             padding: 5px !important;
#             border-radius: 50% !important;
#             width: 30px !important;
#             height: 30px !important;
#             display: flex !important;
#             align-items: center !important;
#             justify-content: center !important;
#         }
        
#         .close-text-input:hover {
#             background: rgba(0, 0, 0, 0.1) !important;
#         }

#         /* Top left TTS toggle - Fixed positioning */
#         .tts-toggle-container {
#             position: fixed;
#             top: 20px;
#             left: 20px;
#             z-index: 1000;
#             background: #2c3e50;
#             border-radius: 8px;
#             padding: 10px 15px;
#             display: flex;
#             align-items: center;
#             gap: 10px;
#             box-shadow: 0 2px 8px rgba(0,0,0,0.2);
#             color: white;
#         }
        
#         .tts-toggle-container .tts-label {
#             color: white;
#             font-weight: 500;
#             font-size: 14px;
#             margin: 0;
#         }
        
#         .tts-toggle-container .stCheckbox {
#             margin: 0;
#         }
        
#         .tts-toggle-container .stCheckbox > div {
#             margin: 0;
#         }
        
#         .tts-toggle-container .stCheckbox label {
#             color: white !important;
#             font-size: 14px !important;
#             margin: 0 !important;
#         }
        
#         .tts-toggle-container .stCheckbox input[type="checkbox"] {
#             margin-right: 8px;
#         }
        
#         /* Add some bottom padding to main content to avoid overlap with fixed input */
#         .main-content {
#             padding-bottom: 120px;
#         }
        
#         /* Hide default streamlit elements that might interfere */
#         .stCheckbox > div > div {
#             display: flex !important;
#             align-items: center !important;
#         }

#     </style>
    
#     <script>
#     document.addEventListener('DOMContentLoaded', function() {
#         // Smooth scroll to bottom of chat
#         function scrollToBottom() {
#             const chatContainer = document.querySelector('[data-testid="stVerticalBlock"]');
#             if (chatContainer) {
#                 chatContainer.scrollTop = chatContainer.scrollHeight;
#             }
#         }
        
#         // Scroll to bottom when new message arrives
#         const chatObserver = new MutationObserver(scrollToBottom);
#         const config = { childList: true, subtree: true };
#         const targetChat = document.querySelector('[data-testid="stVerticalBlock"]');
#         if (targetChat) {
#             chatObserver.observe(targetChat, config);
#         }

#         // Add custom hover effects to buttons
#         const buttons = document.querySelectorAll('.stButton button');
#         buttons.forEach(btn => {
#             btn.addEventListener('mouseenter', () => {
#                 btn.style.transform = 'translateY(-2px)';
#                 btn.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
#             });
#             btn.addEventListener('mouseleave', () => {
#                 btn.style.transform = '';
#                 btn.style.boxShadow = '';
#             });
#         });
        
#         // Add custom class to chat messages based on role
#         const messageObserver = new MutationObserver(function(mutations) {
#             mutations.forEach(function(mutation) {
#                 if (mutation.addedNodes) {
#                     mutation.addedNodes.forEach(function(node) {
#                         if (node.nodeType === 1 && node.hasAttribute('data-testid') && node.getAttribute('data-testid') === 'stChatMessage') {
#                             const roleImg = node.querySelector('[role="img"]');
#                             if (roleImg) {
#                                 const role = roleImg.getAttribute('aria-label');
#                                 if (role === 'user') {
#                                     node.classList.add('user');
#                                 } else if (role === 'assistant') {
#                                     node.classList.add('assistant');
#                                 }
#                             }
#                         }
#                     });
#                 }
#             });
#         });
#         const chatContainerForMessages = document.querySelector('[data-testid="stVerticalBlock"]');
#         if (chatContainerForMessages) {
#             messageObserver.observe(chatContainerForMessages, { childList: true, subtree: true });
#         }

#         // Apply classes to existing messages on page load
#         document.querySelectorAll('[data-testid="stChatMessage"]').forEach(msg => {
#             const role = msg.querySelector('[role="img"]')?.getAttribute('aria-label');
#             if (role === 'user') {
#                 msg.classList.add('user');
#             } else if (role === 'assistant') {
#                 msg.classList.add('assistant');
#             }
#         });
#     });
#     </script>
#     """, unsafe_allow_html=True)

#     st.title("🧠 Smart Excel Data Chat")
#     st.markdown("Upload Excel data and ask intelligent questions - now with voice-first interface!")

#     # Initialize session state for TTS preference
#     if "enable_tts" not in st.session_state:
#         st.session_state["enable_tts"] = True # TTS enabled by default

#     # Top-left TTS toggle button - Fixed implementation
#     tts_toggle_col1, tts_toggle_col2 = st.columns([1, 10])
    
#     with tts_toggle_col1:
#         st.markdown(
#             """
#             <div class="tts-toggle-container">
#                 <span class="tts-label">🔊 Voice Insights</span>
#             </div>
#             """, unsafe_allow_html=True
#         )
        
#         # Create the toggle checkbox
#         tts_enabled = st.checkbox(
#             "Enable TTS", 
#             value=st.session_state["enable_tts"], 
#             key="tts_main_toggle",
#             help="Enable/Disable voice responses"
#         )
        
#         # Update session state if changed
#         if tts_enabled != st.session_state["enable_tts"]:
#             st.session_state["enable_tts"] = tts_enabled

#     # TTS Introduction - only if TTS is enabled
#     if "tts_intro_done" not in st.session_state:
#         if st.session_state["enable_tts"]:
#             speak_command = "Hi, I'm your in-house sales analyzer. Please upload your Excel file to begin."
#             speak_with_azure(speak_command)
#         st.session_state["tts_intro_done"] = True

#     # Initialize other session state variables
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []
#     if 'context' not in st.session_state:
#         st.session_state.context = ConversationContext()
#     if 'execution_timer' not in st.session_state:
#         st.session_state.execution_timer = ExecutionTimer()
#     if 'current_file_hash' not in st.session_state:
#         st.session_state.current_file_hash = ""
#     if 'current_df_hash' not in st.session_state:
#         st.session_state.current_df_hash = ""
#     if 'cached_df' not in st.session_state:
#         st.session_state.cached_df = None
#     if 'cached_conn' not in st.session_state:
#         st.session_state.cached_conn = None
#     if 'tts_query_ready_done' not in st.session_state:
#         st.session_state.tts_query_ready_done = False
#     if 'input_mode' not in st.session_state:
#         st.session_state.input_mode = "voice" # Default to voice mode
#     if 'show_text_input' not in st.session_state:
#         st.session_state.show_text_input = False

#     # Sidebar
#     with st.sidebar:
#         st.header("📂 Upload Data")
#         uploaded_file = st.file_uploader(
#             "Choose Excel file",
#             type=['xlsx', 'xls'],
#             help="Upload your Excel file to start intelligent data chat"
#         )

#         # Data loading with caching
#         if uploaded_file is not None:
#             try:
#                 new_file_hash = get_file_hash(uploaded_file)

#                 if new_file_hash != st.session_state.current_file_hash:
#                     st.session_state.current_file_hash = new_file_hash

#                     excel_file = pd.ExcelFile(uploaded_file)
#                     sheet_name = st.selectbox("📋 Select Sheet:", excel_file.sheet_names)

#                     df, error = load_excel_file(uploaded_file, sheet_name, new_file_hash)

#                     if error:
#                         st.error(f"❌ Error loading file: {error}")
#                     elif df is not None and not df.empty:
#                         new_df_hash = get_dataframe_hash(df)

#                         if new_df_hash != st.session_state.current_df_hash:
#                             st.session_state.current_df_hash = new_df_hash
#                             st.session_state.cached_df = df

#                             if hasattr(st.session_state, 'cached_conn') and st.session_state.cached_conn:
#                                 try:
#                                     st.session_state.cached_conn.close()
#                                 except:
#                                     pass
#                             st.session_state.cached_conn = create_sqlite_db(df)

#                             st.session_state.context = ConversationContext()
#                             st.session_state.context.data_summary = f"Data loaded with {len(df)} rows and {len(df.columns)} columns"
#                             st.session_state.messages = []

#                             st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
#                             if not st.session_state.tts_query_ready_done and st.session_state["enable_tts"]:
#                                 speak_with_azure("Great! Your data is ready. You can now speak your query using the microphone.")
#                                 st.session_state.tts_query_ready_done = True
#                         else:
#                             st.info("📊 Using cached data (no changes detected)")
#                             df = st.session_state.cached_df
#                 else:
#                     if st.session_state.cached_df is not None:
#                         excel_file = pd.ExcelFile(uploaded_file)
#                         sheet_name = st.selectbox("📋 Select Sheet:", excel_file.sheet_names)
#                         df = st.session_state.cached_df
#                         st.info("📊 Using cached data")
#                     else:
#                         df = None

#                 if st.session_state.cached_df is not None:
#                     df = st.session_state.cached_df

#                     with st.expander("📊 Data Preview"):
#                         st.dataframe(df.head())

#                     with st.expander("📋 Column Info"):
#                         col_info = []
#                         for col in df.columns:
#                             col_type = str(df[col].dtype)
#                             null_count = df[col].isnull().sum()
#                             unique_count = df[col].nunique()
#                             col_info.append({
#                                 'Column': col,
#                                 'Type': col_type,
#                                 'Null Values': null_count,
#                                 'Unique Values': unique_count
#                             })
#                         st.dataframe(pd.DataFrame(col_info))

#                     with st.expander("📈 Data Statistics"):
#                         st.write(f"**Total Rows:** {len(df):,}")
#                         st.write(f"**Total Columns:** {len(df.columns)}")
#                         st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

#                         numeric_cols = df.select_dtypes(include=['number']).columns
#                         categorical_cols = df.select_dtypes(include=['object']).columns

#                         if len(numeric_cols) > 0:
#                             st.write(f"**Numeric Columns:** {len(numeric_cols)}")
#                         if len(categorical_cols) > 0:
#                             st.write(f"**Categorical Columns:** {len(categorical_cols)}")

#             except Exception as e:
#                 st.error(f"❌ Error processing file: {str(e)}")

#         # Clear buttons
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("🔄 Clear Chat"):
#                 st.session_state.messages = []
#                 st.session_state.context = ConversationContext()
#                 for key in list(st.session_state.keys()):
#                     if key.startswith("show_viz_"):
#                         del st.session_state[key]
#                 st.rerun()

#         with col2:
#             if st.button("🗑️ Clear Cache"):
#                 st.session_state.current_file_hash = ""
#                 st.session_state.current_df_hash = ""
#                 st.session_state.cached_df = None
#                 if st.session_state.cached_conn:
#                     st.session_state.cached_conn.close()
#                 st.session_state.cached_conn = None
#                 st.session_state.messages = []
#                 st.session_state.context = ConversationContext()
#                 st.cache_data.clear()
#                 st.cache_resource.clear()
#                 st.success("🧹 Cache cleared!")
#                 st.rerun()

#         # AI Configuration
#         st.header("🤖 AI Setup")
#         client = initialize_azure_client()

#         # Performance Info
#         if st.session_state.cached_df is not None:
#             st.header("⚡ Performance")
#             st.success("✅ Data cached in memory")
#             st.success("✅ Database connection ready")
#             if client:
#                 st.success("✅ AI client initialized")

#         # Example queries
#         st.header("💡 Example Queries")
#         st.markdown("""
#         - Show me the top 10 records by sales
#         - What's the average price by category?
#         - Create a chart showing trends over time
#         """)
        
#         # Input Mode Selection
#         st.header("⌨️ Input Mode")
#         input_mode = st.radio(
#             "Choose input method:",
#             ["Voice (Default)", "Text Input"],
#             index=0 if st.session_state.input_mode == "voice" else 1,
#             help="Select how you want to interact with your data"
#         )
        
#         # Update input mode based on selection
#         if input_mode == "Voice (Default)":
#             st.session_state.input_mode = "voice"
#             st.session_state.show_text_input = False
#         else:
#             st.session_state.input_mode = "text"
#             st.session_state.show_text_input = True
    
#     # Main interface with proper spacing
#     st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
#     if st.session_state.cached_df is None:
#         st.info("👆 Please upload an Excel file to start chatting with your data!")
#         st.markdown('</div>', unsafe_allow_html=True)
#         return

#     if client is None:
#         st.error("❌ Please configure Azure OpenAI credentials in the sidebar")
#         st.markdown('</div>', unsafe_allow_html=True)
#         return

#     # Display chat history
#     chat_history_container = st.container()
#     with chat_history_container:
#         for i, message in enumerate(st.session_state.messages):
#             with st.chat_message(message["role"]):
#                 if message["role"] == "assistant":
#                     # Display insights first
#                     st.write(message["content"])

#                     # Display visualization button and chart if present
#                     if "figure" in message and message["figure"] is not None:
#                         viz_key = f"viz_{i}"
#                         if st.button("📊 Show Chart", key=viz_key):
#                             st.session_state[f"show_viz_{i}"] = not st.session_state.get(f"show_viz_{i}", False)

#                         if st.session_state.get(f"show_viz_{i}", False):
#                             if isinstance(message["figure"], str):
#                                 st.error(message["figure"])
#                             else:
#                                 st.plotly_chart(message["figure"], use_container_width=True)

#                     # Display dataframe if present
#                     if "dataframe" in message and not message["dataframe"].empty:
#                         if len(message["dataframe"]) <= 20:
#                             st.dataframe(message["dataframe"], use_container_width=True)
#                         else:
#                             st.write(f"*Showing first 10 of {len(message['dataframe'])} results:*")
#                             st.dataframe(message["dataframe"].head(10), use_container_width=True)
#                 else:
#                     st.write(message["content"])
    
#     st.markdown('</div>', unsafe_allow_html=True)

#     # Determine mic position based on whether there are messages
#     has_messages = len(st.session_state.messages) > 0
#     mic_container_class = "voice-input-container-right" if has_messages else "voice-first-input-container"
    
#     # Voice input interface - position changes based on message history
#     if not has_messages:
#         st.markdown(f'''
#         <div class="{mic_container_class}">
#             <div class="ask-data-text">Ask about your data</div>
#         </div>
#         ''', unsafe_allow_html=True)

#         # Create centered microphone button when no messages
#         col1, col2, col3 = st.columns([1, 1, 1])
#         with col2:  # Center column for main mic button
#             # Try to load custom mic image, fallback to emoji
#             try:
#                 mic_button_html = f'''
#                 <button class="main-mic-button" onclick="document.getElementById('main_mic_btn').click()">
#                      alt="Microphone" class="mic-image" onerror="this.style.display='none'; this.parentElement.innerHTML='🎙️';">
#                 </button>
#                 '''
#                 st.markdown(mic_button_html, unsafe_allow_html=True)
#             except:
#                 pass  # Fallback to regular button
                
#             if st.button("🎙️", key="main_mic_btn", help="Click to speak your query"):
#                 with st.spinner("🎙️ Listening..."):
#                     voice_text = transcribe_speech()
#                     if voice_text:
#                         st.session_state.messages.append({"role": "user", "content": voice_text})
#                         st.rerun()
#     else:
#         # Show mic in bottom right corner when messages exist
#         st.markdown(f'''
#         <div class="{mic_container_class}">
#             <div class="ask-data-text">Ask more</div>
#         </div>
#         ''', unsafe_allow_html=True)
        
#         # Create floating mic button in bottom right
#         mic_col1, mic_col2, mic_col3, mic_col4, mic_col5 = st.columns([1, 1, 1, 1, 1])
#         with mic_col5:  # Rightmost column
#             # Try to load custom mic image, fallback to emoji
#             try:
#                 mic_button_html = f'''
#                 <button class="main-mic-button" onclick="document.getElementById('corner_mic_btn').click()">
#                     <img src="logo/mic_logo.png" alt="Microphone" class="mic-image" onerror="this.style.display='none'; this.parentElement.innerHTML='🎙️';">
#                 </button>
#                 '''
#                 st.markdown(mic_button_html, unsafe_allow_html=True)
#             except:
#                 pass  # Fallback to regular button
                
#             if st.button("🎙️", key="corner_mic_btn", help="Click to speak your query"):
#                 with st.spinner("🎙️ Listening..."):
#                     voice_text = transcribe_speech()
#                     if voice_text:
#                         st.session_state.messages.append({"role": "user", "content": voice_text})
#                         st.rerun()

#     # Text input overlay - only show if text mode is selected
#     if st.session_state.show_text_input:
#         st.markdown('<div class="text-input-overlay">', unsafe_allow_html=True)
        
#         # Close button
#         if st.button("✕", key="close_text_input", help="Close text input"):
#             st.session_state.show_text_input = False
#             st.session_state.input_mode = "voice"
#             st.rerun()
        
#         # Text input
#         user_input = st.text_input("Type your question here...", key="text_input_overlay", placeholder="Ask about your data...")
        
#         if user_input:
#             st.session_state.messages.append({"role": "user", "content": user_input})
#             st.session_state.show_text_input = False
#             st.rerun()
        
#         st.markdown('</div>', unsafe_allow_html=True)

#     # Process the latest message if it's from user
#     if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
#         prompt = st.session_state.messages[-1]["content"]
        
#         with st.chat_message("assistant"):
#             with st.spinner("🤔 Analyzing your data..."):
#                 if not is_data_related_query(prompt):
#                     response = """I'm your data analysis assistant! I can help you explore and understand your data.

# Try asking questions like:
# - "Show me the top 10 records by [column name]"
# - "What's the average/sum/count of [column]?"
# - "Evolution of [column] over time"
# - "Compare [column] by [category]"
# - "Find records where [condition]"
# - "Tell me more about [entity from previous results]"
# - "Explain the previous results in more detail"

# What would you like to know about your data?"""

#                     st.write(response)
#                     # Only speak if TTS is enabled
#                     if st.session_state["enable_tts"]:
#                         speak_with_azure(response)
#                     st.session_state.messages.append({"role": "assistant", "content": response})
#                     return

#                 try:
#                     if not st.session_state.cached_conn:
#                         st.error("❌ Database connection not available. Please reload your data.")
#                         return

#                     insights, result_df, fig = process_user_query(
#                         prompt,
#                         client,
#                         st.session_state.cached_conn,
#                         st.session_state.context
#                     )

#                     with st.expander("⏱️ Performance Metrics"):
#                         st.code(st.session_state.execution_timer.display_timings())

#                     # Display results
#                     if fig:
#                         st.plotly_chart(fig, use_container_width=True)
#                     st.write(insights)
                    
#                     # Only speak if TTS is enabled
#                     if st.session_state["enable_tts"]:
#                         speak_with_azure(insights)

#                     if not result_df.empty:
#                         if len(result_df) <= 20:
#                             st.dataframe(result_df, use_container_width=True)

#                         else:
#                             st.write(f"*Showing first 10 of {len(result_df)} results:*")
#                             st.dataframe(result_df.head(10), use_container_width=True)

#                     # Store the complete response in session state
#                     assistant_message = {
#                         "role": "assistant", 
#                         "content": insights,
#                         "dataframe": result_df,
#                         "figure": fig
#                     }
#                     st.session_state.messages.append(assistant_message)

#                 except Exception as e:
#                     error_msg = f"⚠️ I encountered an issue processing your request: {str(e)}"
#                     st.error(error_msg)
#                     # Only speak if TTS is enabled
#                     if st.session_state["enable_tts"]:
#                         speak_with_azure(error_msg)
#                     st.session_state.messages.append({"role": "assistant", "content": error_msg})


# if __name__ == "__main__":
#     main()

def main():
    # Custom CSS and JavaScript for enhanced UI
    st.markdown("""
    <style>
        /* Main container styling */
        .stApp {
            background-color: #f5f7fa;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #2c3e50 !important;
            color: white !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: white !important;
        }
        
        /* Chat message styling */
        [data-testid="stChatMessage"] {
            padding: 12px 16px;
            border-radius: 12px;
            margin-bottom: 8px;
            max-width: 80%;
        }
        
        [data-testid="stChatMessage"] p {
            margin: 0;
        }
        
        .stChatMessage.user {
            background-color: #e3f2fd;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        
        .stChatMessage.assistant {
            background-color: #f1f1f1;
            margin-right: auto;
            border-bottom-left-radius: 4px;
        }
        
        /* Button styling */
        .stButton button {
            border-radius: 8px !important;
            transition: all 0.3s ease;
        }
        
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Input field styling */
        .stTextInput input {
            border-radius: 12px !important;
            padding: 10px 16px !important;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        
        /* Custom animation for loading */
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        
        .stSpinner > div {
            animation: pulse 1.5s infinite ease-in-out;
        }
        
        /* Text input overlay */
        .text-input-overlay {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(255, 255, 255, 0.98);
            backdrop-filter: blur(10px);
            padding: 20px;
            box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.1);
            z-index: 1001;
            border-top: 1px solid rgba(0, 0, 0, 0.1);
        }
        
        .text-input-overlay .stTextInput {
            margin-bottom: 0;
        }
        
        .text-input-overlay .stTextInput input {
            border: 2px solid #667eea !important;
            padding: 12px 16px !important;
            font-size: 16px !important;
        }
        
        /* Close button for text input overlay */
        .close-text-input {
            position: absolute;
            top: 10px;
            right: 15px;
            background: none !important;
            border: none !important;
            font-size: 20px !important;
            color: #666 !important;
            cursor: pointer !important;
            padding: 5px !important;
            border-radius: 50% !important;
            width: 30px !important;
            height: 30px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        .close-text-input:hover {
            background: rgba(0, 0, 0, 0.1) !important;
        }

        /* Top left TTS toggle - Fixed positioning */
        .tts-toggle-container {
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 1000;
            background: #2c3e50;
            border-radius: 8px;
            padding: 10px 15px;
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            color: white;
        }
        
        .tts-toggle-container .tts-label {
            color: white;
            font-weight: 500;
            font-size: 14px;
            margin: 0;
        }
        
        /* Fixed microphone button */
        .fixed-mic-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
        }
        

        
        .fixed-mic-button:hover {
            transform: scale(1.1) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
        }
        


        /* Add some bottom padding to main content to avoid overlap with chat */
        .main-content {
            padding-bottom: 100px;
        }
        
        /* Hide the actual Streamlit button */
        .fixed-mic-btn-hidden {
            display: none !important;
        }

    </style>
    
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        // Smooth scroll to bottom of chat
        function scrollToBottom() {
            const chatContainer = document.querySelector('[data-testid="stVerticalBlock"]');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }
        
        // Scroll to bottom when new message arrives
        const chatObserver = new MutationObserver(scrollToBottom);
        const config = { childList: true, subtree: true };
        const targetChat = document.querySelector('[data-testid="stVerticalBlock"]');
        if (targetChat) {
            chatObserver.observe(targetChat, config);
        }

        // Add custom hover effects to buttons
        const buttons = document.querySelectorAll('.stButton button');
        buttons.forEach(btn => {
            btn.addEventListener('mouseenter', () => {
                btn.style.transform = 'translateY(-2px)';
                btn.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
            });
            btn.addEventListener('mouseleave', () => {
                btn.style.transform = '';
                btn.style.boxShadow = '';
            });
        });
        
        // Add custom class to chat messages based on role
        const messageObserver = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
                if (mutation.addedNodes) {
                    mutation.addedNodes.forEach(function(node) {
                        if (node.nodeType === 1 && node.hasAttribute('data-testid') && node.getAttribute('data-testid') === 'stChatMessage') {
                            const roleImg = node.querySelector('[role="img"]');
                            if (roleImg) {
                                const role = roleImg.getAttribute('aria-label');
                                if (role === 'user') {
                                    node.classList.add('user');
                                } else if (role === 'assistant') {
                                    node.classList.add('assistant');
                                }
                            }
                        }
                    });
                }
            });
        });
        const chatContainerForMessages = document.querySelector('[data-testid="stVerticalBlock"]');
        if (chatContainerForMessages) {
            messageObserver.observe(chatContainerForMessages, { childList: true, subtree: true });
        }

        // Apply classes to existing messages on page load
        document.querySelectorAll('[data-testid="stChatMessage"]').forEach(msg => {
            const role = msg.querySelector('[role="img"]')?.getAttribute('aria-label');
            if (role === 'user') {
                msg.classList.add('user');
            } else if (role === 'assistant') {
                msg.classList.add('assistant');
            }
        });
    });
    </script>
    """, unsafe_allow_html=True)

    st.title("🧠 Smart Excel Data Chat")
    st.markdown("Upload Excel data and ask intelligent questions - now with voice-first interface!")

    # Initialize session state for TTS preference
    if "enable_tts" not in st.session_state:
        st.session_state["enable_tts"] = True # TTS enabled by default
    
    # Initialize mic recording state
    if "mic_recording" not in st.session_state:
        st.session_state["mic_recording"] = False

    # Top-left TTS toggle button - Fixed implementation
    st.markdown(
        """
        <div class="tts-toggle-container">
            <span class="tts-label">🔊 Voice Insights</span>
        </div>
        """, unsafe_allow_html=True
    )
    
    # Create the toggle checkbox
    tts_enabled = st.checkbox(
        "Enable TTS", 
        value=st.session_state["enable_tts"], 
        key="tts_main_toggle",
        help="Enable/Disable voice responses"
    )
    
    # Update session state if changed
    if tts_enabled != st.session_state["enable_tts"]:
        st.session_state["enable_tts"] = tts_enabled

    # TTS Introduction - only if TTS is enabled
    if "tts_intro_done" not in st.session_state:
        if st.session_state["enable_tts"]:
            speak_command = "Hi, I'm your in-house sales analyzer. Please upload your Excel file to begin."
            speak_with_azure(speak_command)
        st.session_state["tts_intro_done"] = True

    # Initialize other session state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'context' not in st.session_state:
        st.session_state.context = ConversationContext()
    if 'execution_timer' not in st.session_state:
        st.session_state.execution_timer = ExecutionTimer()
    if 'current_file_hash' not in st.session_state:
        st.session_state.current_file_hash = ""
    if 'current_df_hash' not in st.session_state:
        st.session_state.current_df_hash = ""
    if 'cached_df' not in st.session_state:
        st.session_state.cached_df = None
    if 'cached_conn' not in st.session_state:
        st.session_state.cached_conn = None
    if 'tts_query_ready_done' not in st.session_state:
        st.session_state.tts_query_ready_done = False
    if 'input_mode' not in st.session_state:
        st.session_state.input_mode = "voice" # Default to voice mode
    if 'show_text_input' not in st.session_state:
        st.session_state.show_text_input = False

    # Sidebar
    with st.sidebar:
        st.header("📂 Upload Data")
        uploaded_file = st.file_uploader(
            "Choose Excel file",
            type=['xlsx', 'xls'],
            help="Upload your Excel file to start intelligent data chat"
        )

        # Data loading with caching
        if uploaded_file is not None:
            try:
                new_file_hash = get_file_hash(uploaded_file)

                if new_file_hash != st.session_state.current_file_hash:
                    st.session_state.current_file_hash = new_file_hash

                    excel_file = pd.ExcelFile(uploaded_file)
                    sheet_name = st.selectbox("📋 Select Sheet:", excel_file.sheet_names)

                    df, error = load_excel_file(uploaded_file, sheet_name, new_file_hash)

                    if error:
                        st.error(f"❌ Error loading file: {error}")
                    elif df is not None and not df.empty:
                        new_df_hash = get_dataframe_hash(df)

                        if new_df_hash != st.session_state.current_df_hash:
                            st.session_state.current_df_hash = new_df_hash
                            st.session_state.cached_df = df

                            if hasattr(st.session_state, 'cached_conn') and st.session_state.cached_conn:
                                try:
                                    st.session_state.cached_conn.close()
                                except:
                                    pass
                            st.session_state.cached_conn = create_sqlite_db(df)

                            st.session_state.context = ConversationContext()
                            st.session_state.context.data_summary = f"Data loaded with {len(df)} rows and {len(df.columns)} columns"
                            st.session_state.messages = []

                            st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
                            if not st.session_state.tts_query_ready_done and st.session_state["enable_tts"]:
                                speak_with_azure("Great! Your data is ready. You can now speak your query using the microphone.")
                            st.session_state.tts_query_ready_done = True
                        else:
                            st.info("📊 Using cached data (no changes detected)")
                            df = st.session_state.cached_df
                else:
                    if st.session_state.cached_df is not None:
                        excel_file = pd.ExcelFile(uploaded_file)
                        sheet_name = st.selectbox("📋 Select Sheet:", excel_file.sheet_names)
                        df = st.session_state.cached_df
                        st.info("📊 Using cached data")
                    else:
                        df = None

                if st.session_state.cached_df is not None:
                    df = st.session_state.cached_df

                    with st.expander("📊 Data Preview"):
                        st.dataframe(df.head())

                    with st.expander("📋 Column Info"):
                        col_info = []
                        for col in df.columns:
                            col_type = str(df[col].dtype)
                            null_count = df[col].isnull().sum()
                            unique_count = df[col].nunique()
                            col_info.append({
                                'Column': col,
                                'Type': col_type,
                                'Null Values': null_count,
                                'Unique Values': unique_count
                            })
                        st.dataframe(pd.DataFrame(col_info))

                    with st.expander("📈 Data Statistics"):
                        st.write(f"**Total Rows:** {len(df):,}")
                        st.write(f"**Total Columns:** {len(df.columns)}")
                        st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

                        numeric_cols = df.select_dtypes(include=['number']).columns
                        categorical_cols = df.select_dtypes(include=['object']).columns

                        if len(numeric_cols) > 0:
                            st.write(f"**Numeric Columns:** {len(numeric_cols)}")
                        if len(categorical_cols) > 0:
                            st.write(f"**Categorical Columns:** {len(categorical_cols)}")

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")

        # Clear buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear Chat"):
                st.session_state.messages = []
                st.session_state.context = ConversationContext()
                for key in list(st.session_state.keys()):
                    if key.startswith("show_viz_"):
                        del st.session_state[key]
                st.rerun()

        with col2:
            if st.button("🗑️ Clear Cache"):
                st.session_state.current_file_hash = ""
                st.session_state.current_df_hash = ""
                st.session_state.cached_df = None
                if st.session_state.cached_conn:
                    st.session_state.cached_conn.close()
                st.session_state.cached_conn = None
                st.session_state.messages = []
                st.session_state.context = ConversationContext()
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("🧹 Cache cleared!")
                st.rerun()

        # AI Configuration
        st.header("🤖 AI Setup")
        client = initialize_azure_client()

        # Performance Info
        if st.session_state.cached_df is not None:
            st.header("⚡ Performance")
            st.success("✅ Data cached in memory")
            st.success("✅ Database connection ready")
            if client:
                st.success("✅ AI client initialized")

        # Example queries
        st.header("💡 Example Queries")
        st.markdown("""
        - Show me the top 10 records by sales
        - What's the average price by category?
        - Create a chart showing trends over time
        """)
        
        # Input Mode Selection
        st.header("⌨️ Input Mode")
        input_mode = st.radio(
            "Choose input method:",
            ["Voice (Default)", "Text Input"],
            index=0 if st.session_state.input_mode == "voice" else 1,
            help="Select how you want to interact with your data"
        )
        
        # Update input mode based on selection
        if input_mode == "Voice (Default)":
            st.session_state.input_mode = "voice"
            st.session_state.show_text_input = False
        else:
            st.session_state.input_mode = "text"
            st.session_state.show_text_input = True
    
    # Main interface with proper spacing
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    if st.session_state.cached_df is None:
        st.info("👆 Please upload an Excel file to start chatting with your data!")
        st.markdown('</div>', unsafe_allow_html=True) # Close main-content
        return

    if client is None:
        st.error("❌ Please configure Azure OpenAI credentials in the sidebar")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # Display chat history
    chat_history_container = st.container()
    with chat_history_container:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    # Display insights first
                    st.write(message["content"])

                    # Display visualization button and chart if present
                    if "figure" in message and message["figure"] is not None:
                        viz_key = f"viz_{i}"
                        # Ensure the button toggles visibility, not reruns immediately
                        if st.button("📊 Show Chart", key=viz_key, on_click=lambda k=viz_key: st.session_state.update({k: not st.session_state.get(k, False)})):
                            pass # The state update is handled by on_click

                        if st.session_state.get(f"show_viz_{i}", False):
                            if isinstance(message["figure"], str):
                                st.error(message["figure"])
                            else:
                                st.plotly_chart(message["figure"], use_container_width=True)

                    # Display dataframe if present
                    if "dataframe" in message and not message["dataframe"].empty:
                        if len(message["dataframe"]) <= 20:
                            st.dataframe(message["dataframe"], use_container_width=True)
                        else:
                            st.write(f"*Showing first 10 of {len(message['dataframe'])} results:*")
                            st.dataframe(message["dataframe"].head(10), use_container_width=True)
                else:
                    st.write(message["content"])
    
    st.markdown('</div>', unsafe_allow_html=True) # Close main-content

    # Text input overlay - only show if text mode is selected
    if st.session_state.show_text_input:
        st.markdown('<div class="text-input-overlay">', unsafe_allow_html=True)
        
        # Close button for the text input overlay
        st.markdown("""
        <button class="close-text-input" onclick="document.getElementById('close_text_input_btn').click()">✕</button>
        """, unsafe_allow_html=True)
        # A hidden Streamlit button to capture the click
        if st.button("✕", key="close_text_input_btn", help="Close text input"):
            st.session_state.show_text_input = False
            st.session_state.input_mode = "voice" # Revert to voice mode
            st.rerun()
        
        # Text input field
        user_input = st.text_input("Type your question here...", key="text_input_overlay_field", placeholder="Ask about your data...")
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            # After input, usually you'd want the overlay to disappear or switch back to voice mode
            st.session_state.show_text_input = False 
            st.session_state.input_mode = "voice"
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Fixed microphone button in bottom right corner
    # Update the button's class based on recording state
    mic_class = "fixed-mic-button recording" if st.session_state.mic_recording else "fixed-mic-button"

    
    # Hidden Streamlit button that gets triggered by the fixed mic - make it completely invisible

        

    if st.button("🎙️", key="fixed_mic_btn", help="Click to speak your query", type="primary"):
        st.session_state.input_mode = "voice"
        st.session_state.mic_recording = True
        st.rerun()

    # Add CSS for animation when recording

    # Handle microphone recording
    if st.session_state.mic_recording:
        with st.spinner("🎙️ Listening..."):
            voice_text = transcribe_speech()
            st.session_state.mic_recording = False
            if voice_text:
                st.session_state.messages.append({"role": "user", "content": voice_text})
                st.rerun()
            else:
                st.rerun()  # Just rerun to reset the recording state

    # Process the latest message if it's from user
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        prompt = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your data..."):
                if not is_data_related_query(prompt):
                    response = """I'm your data analysis assistant! I can help you explore and understand your data.

Try asking questions like:
- "Show me the top 10 records by [column name]"
- "What's the average/sum/count of [column]?"
- "Evolution of [column] over time"
- "Compare [column] by [category]"
- "Find records where [condition]"
- "Tell me more about [entity from previous results]"
- "Explain the previous results in more detail"

What would you like to know about your data?"""

                    st.write(response)
                    # Only speak if TTS is enabled
                    if st.session_state["enable_tts"]:
                        speak_with_azure(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    return

                try:
                    if not st.session_state.cached_conn:
                        st.error("❌ Database connection not available. Please reload your data.")
                        return

                    insights, result_df, fig = process_user_query(
                        prompt,
                        client,
                        st.session_state.cached_conn,
                        st.session_state.context
                    )

                    with st.expander("⏱️ Performance Metrics"):
                        st.code(st.session_state.execution_timer.display_timings())

                    # Display results
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    st.write(insights)

                    # Only speak if TTS is enabled
                    if st.session_state["enable_tts"]:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            speak_with_azure(insights)
                        with col2:
                            if st.button("Stop Speech", key="stop_speech"):
                                stop_speaking()

                    if not result_df.empty:
                        if len(result_df) <= 20:
                            st.dataframe(result_df, use_container_width=True)

                        else:
                            st.write(f"*Showing first 10 of {len(result_df)} results:*")
                            st.dataframe(result_df.head(10), use_container_width=True)

                    # Store the complete response in session state
                    assistant_message = {
                        "role": "assistant", 
                        "content": insights,
                        "dataframe": result_df,
                        "figure": fig
                    }
                    st.session_state.messages.append(assistant_message)

                except Exception as e:
                    error_msg = f"⚠️ I encountered an issue processing your request: {str(e)}"
                    st.error(error_msg)
                    # Only speak if TTS is enabled
                    if st.session_state["enable_tts"]:
                        speak_with_azure(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
