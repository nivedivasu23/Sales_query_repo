import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
import json
import re
import tempfile
import os
from io import StringIO
import base64
import logging
from datetime import datetime
from dotenv import load_dotenv
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Groq Data Analyzer",
    page_icon="📊",
    layout="wide"
)

def log_step(step_name, status="started", details=None):
    """Enhanced logging function for debugging"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # #if status == "started":
    #     #logger.info(f"🚀 STEP {step_name} - STARTED")
    #     #st.info(f"⏱️ {timestamp} - {step_name} started...")
    # if status == "success":
    #     logger.info(f"✅ STEP {step_name} - SUCCESS")
    #     st.success(f"✅ {timestamp} - {step_name} completed successfully!")
    # elif status == "error":
    #     logger.error(f"❌ STEP {step_name} - ERROR: {details}")
    #     st.error(f"❌ {timestamp} - {step_name} failed: {details}")
    # elif status == "warning":
    #     logger.warning(f"⚠️ STEP {step_name} - WARNING: {details}")
    #     st.warning(f"⚠️ {timestamp} - {step_name}: {details}")
    
    # if details and status in ["success", "info"]:
    #     logger.info(f"Details: {details}")

# Initialize Groq client
@st.cache_resource
def init_groq_client():
    # You need to set your Groq API key here
    env_path = '.env'
    load_dotenv(dotenv_path=env_path)
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    #api_key = st.secrets.get("GROQ_API_KEY", "your_groq_api_key_here")
    return Groq(api_key=groq_api_key)

def call_groq_api(client, prompt, model="qwen-qwq-32b"):
    """Call Groq API with enhanced error handling and logging"""
    try:
        log_step("Groq API Call", "started", f"Model: {model}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4000
        )
        
        result = response.choices[0].message.content
        log_step("Groq API Call", "success", f"Response length: {len(result)} characters")
        logger.debug(f"API Response preview: {result[:200]}...")
        
        return result
    except Exception as e:
        log_step("Groq API Call", "error", str(e))
        return None

def understand_query_and_identify_tables(client, user_query):
    """Step 1: Understand query and identify required tables with enhanced logging"""
    log_step("Query Understanding", "started")
    
    try:
        prompt = f"""
        Analyze this user query with deep understanding of business context, data hierarchy, and table relationships:
        Query: "{user_query}"
        
        Your analysis should:
        1. FIRST understand the business domain and context of the query
        2. THEN identify the hierarchical relationships in the data (e.g., organizational structure, time periods, product categories)
        3. CONSIDER master-detail relationships, lookup tables, and key project structures
        4. IDENTIFY table hierarchy levels (master → development → project → transaction)
        5. FINALLY determine the data structures needed to answer the query
        
        SPECIAL ATTENTION TO TABLE HIERARCHY:
        - Master Tables: Core reference data (customers, products, employees, locations)
        - Development Tables: Intermediate grouping/categorization tables
        - Project Tables: Specific project or initiative data with key identifiers
        - Transaction Tables: Detailed operational data linked to projects/masters
        - Lookup Tables: Code/value mappings and classifications
        
        Provide your response as JSON with these components:
        1. Business Context: What domain/industry this query relates to
        2. Data Hierarchy: The natural hierarchy in the data (e.g., Country > Region > Store)
        3. Table Hierarchy: Master → Development → Project → Detail relationships
        4. Core Entities: The main business entities involved
        5. Key Relationships: Critical foreign key relationships and dependencies
        6. Required Tables: Tables needed with proper relationships and hierarchy levels
        
        FORMAT REQUIREMENTS:
        - Columns should reflect actual business attributes
        - Data types should match real-world usage
        - Include primary/foreign keys where relationships exist
        - Time dimensions should be properly typed (DATE/DATETIME)
        - Specify hierarchy level for each table (master/development/project/transaction/lookup)
        
        Example Response Format:
        {{
            "business_context": "e.g., construction project management with master contractors and development phases",
            "data_hierarchy": [
                "Master Company > Development Phase > Project > Task",
                "Master Customer > Project Contract > Project Milestones",
                "Year > Quarter > Month > Week"
            ],
            "table_hierarchy": {{
                "master_tables": ["master_customers", "master_contractors", "master_materials"],
                "development_tables": ["development_phases", "development_categories", "development_teams"],
                "project_tables": ["projects", "project_assignments", "project_milestones"],
                "transaction_tables": ["project_transactions", "material_usage", "time_entries"],
                "lookup_tables": ["status_codes", "phase_types", "material_categories"]
            }},
            "core_entities": ["Master Data", "Development Phases", "Projects", "Transactions"],
            "key_relationships": [
                {{
                    "relationship_type": "master_to_development",
                    "from_table": "master_customers",
                    "to_table": "development_phases",
                    "key_column": "customer_id",
                    "description": "Development phases are planned for master customers"
                }},
                {{
                    "relationship_type": "development_to_project",
                    "from_table": "development_phases",
                    "to_table": "projects",
                    "key_column": "development_phase_id",
                    "description": "Projects are created under specific development phases"
                }},
                {{
                    "relationship_type": "project_to_transaction",
                    "from_table": "projects",
                    "to_table": "project_transactions",
                    "key_column": "project_id",
                    "description": "All transactions are linked to specific projects"
                }}
            ],
            "required_tables": [
                {{
                    "table_name": "master_customers",
                    "hierarchy_level": "master",
                    "description": "Master customer information - top level entity",
                    "columns": [
                        {{"name": "master_customer_id", "type": "INTEGER", "role": "primary_key"}},
                        {{"name": "customer_code", "type": "TEXT", "role": "key_identifier"}},
                        {{"name": "customer_name", "type": "TEXT", "role": "attribute"}},
                        {{"name": "customer_type", "type": "TEXT", "role": "classification"}},
                        {{"name": "established_date", "type": "DATE", "role": "audit"}}
                    ],
                    "relationships": [
                        {{"with_table": "development_phases", "on_column": "master_customer_id", "relationship_type": "one_to_many"}}
                    ]
                }},
                {{
                    "table_name": "development_phases",
                    "hierarchy_level": "development",
                    "description": "Development phases linking master entities to projects",
                    "columns": [
                        {{"name": "development_phase_id", "type": "INTEGER", "role": "primary_key"}},
                        {{"name": "master_customer_id", "type": "INTEGER", "role": "foreign_key"}},
                        {{"name": "phase_code", "type": "TEXT", "role": "key_identifier"}},
                        {{"name": "phase_name", "type": "TEXT", "role": "attribute"}},
                        {{"name": "phase_start_date", "type": "DATE", "role": "time_dimension"}},
                        {{"name": "phase_status", "type": "TEXT", "role": "status"}}
                    ],
                    "relationships": [
                        {{"with_table": "master_customers", "on_column": "master_customer_id", "relationship_type": "many_to_one"}},
                        {{"with_table": "projects", "on_column": "development_phase_id", "relationship_type": "one_to_many"}}
                    ]
                }},
                {{
                    "table_name": "projects",
                    "hierarchy_level": "project",
                    "description": "Individual projects under development phases",
                    "columns": [
                        {{"name": "project_id", "type": "INTEGER", "role": "primary_key"}},
                        {{"name": "development_phase_id", "type": "INTEGER", "role": "foreign_key"}},
                        {{"name": "project_code", "type": "TEXT", "role": "key_identifier"}},
                        {{"name": "project_name", "type": "TEXT", "role": "attribute"}},
                        {{"name": "project_start_date", "type": "DATE", "role": "time_dimension"}},
                        {{"name": "project_budget", "type": "DECIMAL(15,2)", "role": "measure"}},
                        {{"name": "project_status", "type": "TEXT", "role": "status"}}
                    ],
                    "relationships": [
                        {{"with_table": "development_phases", "on_column": "development_phase_id", "relationship_type": "many_to_one"}},
                        {{"with_table": "project_transactions", "on_column": "project_id", "relationship_type": "one_to_many"}}
                    ]
                }},
                {{
                    "table_name": "project_transactions",
                    "hierarchy_level": "transaction",
                    "description": "Detailed transaction records for projects",
                    "columns": [
                        {{"name": "transaction_id", "type": "INTEGER", "role": "primary_key"}},
                        {{"name": "project_id", "type": "INTEGER", "role": "foreign_key"}},
                        {{"name": "transaction_date", "type": "DATE", "role": "time_dimension"}},
                        {{"name": "transaction_amount", "type": "DECIMAL(10,2)", "role": "measure"}},
                        {{"name": "transaction_type", "type": "TEXT", "role": "classification"}},
                        {{"name": "description", "type": "TEXT", "role": "attribute"}}
                    ],
                    "relationships": [
                        {{"with_table": "projects", "on_column": "project_id", "relationship_type": "many_to_one"}}
                    ]
                }}
            ],
            "analysis_recommendation": {{
                "suggested_approach": "Hierarchical analysis starting from master tables down to transactions",
                "join_strategy": "Start with master tables, join through project keys to detail tables",
                "potential_visualizations": [
                    "Hierarchical tree view of master → project → transaction",
                    "Drill-down reports by project hierarchy",
                    "Master-detail dashboards with project summaries"
                ],
                "key_considerations": [
                    "Maintain referential integrity across hierarchy levels",
                    "Consider data lineage from master to transaction",
                    "Ensure proper handling of project key relationships"
                ]
            }}
        }}
        
        YOUR TASK:
        Provide a similarly structured response for the query: "{user_query}"
        Focus on:
        - Real-world business meaning
        - Proper hierarchical data modeling practices
        - Master-detail relationships and project key structures
        - Actionable analysis recommendations considering table hierarchy
        """
        
        response = call_groq_api(client, prompt)
        if response:
            try:
                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    parsed_json = json.loads(json_match.group())
                    log_step("Query Understanding", "success", f"Identified {len(parsed_json.get('required_tables', []))} required tables")
                    return parsed_json
                else:
                    log_step("Query Understanding", "warning", "No JSON found in response")
            except json.JSONDecodeError as e:
                log_step("Query Understanding", "error", f"JSON parsing failed: {str(e)}")
        else:
            log_step("Query Understanding", "error", "No response from API")
            
    except Exception as e:
        log_step("Query Understanding", "error", str(e))
    
    return None

def clean_column_names(df):
    """Clean column names to ensure SQL compatibility and handle duplicates with logging"""
    log_step("Column Name Cleaning", "started")
    
    try:
        # Store original column names mapping
        original_columns = df.columns.tolist()
        logger.info(f"Original columns: {original_columns}")
        
        # Clean column names for SQL compatibility
        clean_columns = []
        seen_columns = {}
        
        for i, col in enumerate(original_columns):
            # Remove special characters and spaces, replace with underscores
            clean_col = re.sub(r'[^\w]', '_', str(col))
            # Remove multiple underscores
            clean_col = re.sub(r'_+', '_', clean_col)
            # Remove leading/trailing underscores
            clean_col = clean_col.strip('_')
            # Ensure it doesn't start with a number
            if clean_col and clean_col[0].isdigit():
                clean_col = 'col_' + clean_col
            # Ensure it's not empty
            if not clean_col:
                clean_col = f'column_{i}'
            
            # Handle duplicates
            if clean_col in seen_columns:
                seen_columns[clean_col] += 1
                clean_col = f"{clean_col}_{seen_columns[clean_col]}"
            else:
                seen_columns[clean_col] = 1
                
            clean_columns.append(clean_col)
        
        # Create mapping dictionary
        column_mapping = dict(zip(clean_columns, original_columns))
        
        # Rename DataFrame columns
        df.columns = clean_columns
        
        logger.info(f"Cleaned columns: {clean_columns}")
        logger.info(f"Column mapping created: {len(column_mapping)} mappings")
        
        log_step("Column Name Cleaning", "success", f"Processed {len(original_columns)} columns")
        return df, column_mapping
        
    except Exception as e:
        log_step("Column Name Cleaning", "error", str(e))
        return df, {}

def excel_to_sqlite(uploaded_file, table_structure):
    """Step 2: Convert Excel to SQLite database with improved column handling and logging"""
    log_step("Excel to SQLite Conversion", "started")
    
    try:
        # Read Excel file
        logger.info(f"Reading file: {uploaded_file.name}")
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        logger.info(f"File loaded successfully. Shape: {df.shape}")
        
        if df.empty:
            log_step("Excel to SQLite Conversion", "error", "DataFrame is empty")
            return None, None, None, None
        
        # Clean column names for SQL compatibility
        df, column_mapping = clean_column_names(df)
        
        # Create temporary SQLite database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db_path = temp_db.name
        temp_db.close()  # Close the file handle so SQLite can open it
        
        logger.info(f"Created temporary database: {temp_db_path}")
        
        conn = sqlite3.connect(temp_db_path)
        
        # If we have table structure, use the first table name, otherwise use 'data'
        table_name = 'data'
        if table_structure and 'required_tables' in table_structure:
            table_name = table_structure['required_tables'][0]['table_name']
        
        logger.info(f"Using table name: {table_name}")
        
        # Store data in SQLite
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        logger.info(f"Data stored in SQLite table '{table_name}'")
        
        conn.close()
        
        log_step("Excel to SQLite Conversion", "success", f"Created table '{table_name}' with {len(df)} rows")
        return temp_db_path, df, table_name, column_mapping
        
    except Exception as e:
        log_step("Excel to SQLite Conversion", "error", str(e))
        return None, None, None, None

def inspect_table_schema(db_path, table_name):
    """Get actual table schema from SQLite database with logging"""
    log_step("Schema Inspection", "started")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table schema
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        logger.info(f"Found {len(columns_info)} columns in table")
        
        # Get sample data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        sample_data = cursor.fetchall()
        logger.info(f"Retrieved {len(sample_data)} sample rows")
        
        conn.close()
        
        # Format column information
        columns = [{'name': col[1], 'type': col[2]} for col in columns_info]
        logger.info(f"Column details: {columns}")
        
        log_step("Schema Inspection", "success", f"Inspected {len(columns)} columns")
        return columns, sample_data
        
    except Exception as e:
        log_step("Schema Inspection", "error", str(e))
        return None, None

def generate_sql_query(client, user_query, table_name, db_path, df_info, column_mapping):
    """Step 3: Generate SQL query based on user query and actual table structure with enhanced logging"""
    log_step("SQL Query Generation", "started")
    
    try:
        # Get actual table schema from database
        actual_columns, sample_data = inspect_table_schema(db_path, table_name)
        
        if not actual_columns:
            log_step("SQL Query Generation", "error", "Could not retrieve table schema")
            return None, None
        
        # Get column names and types
        column_names = [col['name'] for col in actual_columns]
        column_details = f"Available columns with types: {actual_columns}"
        
        logger.info(f"Available SQL columns: {column_names}")
        
        # Create detailed column mapping info
        column_info = []
        for clean_col in column_names:
            original_col = column_mapping.get(clean_col, clean_col)
            column_info.append({
                'sql_name': clean_col,
                'original_name': original_col,
                'data_type': df_info[clean_col].dtype.name if clean_col in df_info.columns else 'unknown'
            })
        
        # Get data types and sample values from DataFrame
        df_dtypes = df_info.dtypes.to_dict()
        df_sample = df_info.head(3).to_dict('records')
        
        prompt = f"""
        Generate a SQL query based on the user's request using ONLY the available SQL column names.
        
        User Query: "{user_query}"
        Table Name: {table_name}
        
        COLUMN MAPPING (Use SQL names in query):
        {json.dumps(column_info, indent=2)}
        
        ACTUAL AVAILABLE SQL COLUMN NAMES: {column_names}
        
        Data Types: {df_dtypes}
        
        Sample Data Records:
        {json.dumps(df_sample, indent=2, default=str)}
        
        IMPORTANT RULES:
        1. Use ONLY the SQL column names (sql_name) from the COLUMN MAPPING above 
        2. Column names are case-sensitive - use them exactly as shown in sql_name
        3. Use the table name: {table_name}
        4. Generate a SQL query that best answers the user's question with available data
        5. If the user refers to original column names, map them to the corresponding SQL names
        6. Return ONLY the SQL query, no explanations
        7. Ensure the query is syntactically correct for SQLite
        8. The query MUST be a valid SQL string

       the sql has to be in this where \n is present ```sql\n?|```

        Note: make sure to always include the master development or projects in the data to showcase about what the query is talking about
        
        SQL Query:
        """
        
        response = call_groq_api(client, prompt)
        if response:
            # Extract SQL query (remove any markdown formatting)
            sql_query = re.sub(r'```sql\n?|```\n?|SQL Query:\s*', '', response).strip()
            
            # Additional cleaning to ensure it's a proper SQL string
            sql_query = sql_query.replace('\n', ' ').strip()
            
            # Validate that we have a non-empty string
            if not sql_query or len(sql_query.strip()) == 0:
                log_step("SQL Query Generation", "error", "Generated query is empty")
                return None, None
            
            # Basic validation that it looks like SQL
            if not any(keyword in sql_query.upper() for keyword in ['SELECT', 'FROM']):
                log_step("SQL Query Generation", "error", f"Generated text doesn't look like SQL: {sql_query[:100]}")
                return None, None
            
            logger.info(f"Generated SQL query: {sql_query}")
            log_step("SQL Query Generation", "success", f"Query length: {len(sql_query)} characters")
            return sql_query, column_names
        else:
            log_step("SQL Query Generation", "error", "No response from API")
            return None, None
            
    except Exception as e:
        log_step("SQL Query Generation", "error", str(e))
        return None, None

def validate_and_fix_sql_query(client, sql_query, table_name, actual_columns, user_query, column_mapping):
    """Validate SQL query against actual columns and fix if needed with logging"""
    log_step("SQL Query Validation", "started")
    
    try:
        # Ensure sql_query is a string
        if not isinstance(sql_query, str):
            log_step("SQL Query Validation", "error", f"SQL query is not a string: {type(sql_query)}")
            return None
        
        if not sql_query.strip():
            log_step("SQL Query Validation", "error", "SQL query is empty")
            return None
        
        # Check if all columns in SQL query exist in actual columns
        sql_upper = sql_query.upper()
        missing_columns = []
        
        # Extract potential column names from SQL (basic parsing)
        potential_columns = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', sql_query)
        
        for col in potential_columns:
            if col not in ['SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'ORDER', 'LIMIT', 'ASC', 'DESC', 
                          'AND', 'OR', 'AS', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', table_name.upper()]:
                if col not in actual_columns:
                    missing_columns.append(col)
        
        if missing_columns:
            log_step("SQL Query Validation", "warning", f"Missing columns detected: {missing_columns}")
            
            # Create detailed column mapping for fixing
            column_info = []
            for clean_col in actual_columns:
                original_col = column_mapping.get(clean_col, clean_col)
                column_info.append({
                    'sql_name': clean_col,
                    'original_name': original_col
                })
            
            # Try to fix the query
            fix_prompt = f"""
            The SQL query contains columns that don't exist in the actual data.
            
            Original Query: {sql_query}
            Missing Columns: {missing_columns}
            
            AVAILABLE COLUMNS WITH MAPPING:
            {json.dumps(column_info, indent=2)}
            
            User Intent: "{user_query}"
            Table Name: {table_name}
            
            Please fix the SQL query by:
            1. Replacing missing columns with the closest matching available SQL column names
            2. Use only the sql_name values from the mapping above
            3. Keep the same intent as the original user query
            4. Ensure the query is syntactically correct for SQLite
            5. Return ONLY the corrected SQL query as a string
            Note: make sure to always include the master development or projects in the data to showcase about what the query is talking about
            Return ONLY the corrected SQL query:
            """
            
            fixed_query = call_groq_api(client, fix_prompt)
            if fixed_query:
                fixed_query = re.sub(r'```sql\n?|```\n?', '', fixed_query).strip()
                log_step("SQL Query Validation", "success", "Query automatically fixed")
                return fixed_query
        
        log_step("SQL Query Validation", "success", "Query validation passed")
        return sql_query
        
    except Exception as e:
        log_step("SQL Query Validation", "error", str(e))
        return sql_query

def execute_sql_query(db_path, sql_query):
    """Step 4: Execute SQL query and return results with enhanced logging"""
    log_step("SQL Query Execution", "started")
    
    try:
        # Validate inputs
        if not isinstance(sql_query, str):
            error_msg = f"SQL query must be a string, got {type(sql_query)}: {sql_query}"
            log_step("SQL Query Execution", "error", error_msg)
            return None, error_msg
        
        if not sql_query.strip():
            error_msg = "SQL query is empty or whitespace"
            log_step("SQL Query Execution", "error", error_msg)
            return None, error_msg
        
        logger.info(f"Executing SQL query: {sql_query}")
        
        conn = sqlite3.connect(db_path)
        result_df = pd.read_sql_query(sql_query, conn)
        conn.close()
        
        logger.info(f"Query executed successfully. Result shape: {result_df.shape}")
        log_step("SQL Query Execution", "success", f"Retrieved {len(result_df)} rows, {len(result_df.columns)} columns")
        
        return result_df, None
        
    except Exception as e:
        error_msg = str(e)
        log_step("SQL Query Execution", "error", error_msg)
        return None, error_msg

def restore_original_column_names(result_df, column_mapping):
    """Restore original column names in the result DataFrame with logging"""
    log_step("Column Name Restoration", "started")
    
    try:
        # Create reverse mapping
        reverse_mapping = {v: k for k, v in column_mapping.items()}
        
        # Rename columns back to original names where possible
        new_columns = []
        for col in result_df.columns:
            if col in column_mapping:
                new_columns.append(column_mapping[col])
            else:
                new_columns.append(col)
        
        result_df.columns = new_columns
        logger.info(f"Restored column names: {new_columns}")
        log_step("Column Name Restoration", "success", f"Restored {len(new_columns)} column names")
        
        return result_df
        
    except Exception as e:
        log_step("Column Name Restoration", "error", str(e))
        return result_df



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
        log_step("Subplot Analysis", "info", f"Multiple subplots recommended: {subplot_analysis['subplot_reasons']}")
        
        # Generate subplot code
        subplot_code = generate_multiple_subplot_code(client, user_query, result_df, subplot_analysis, column_mapping)
        
        if subplot_code:
            return subplot_code, True  # Return code and subplot flag
    
    # Fallback to single plot
    log_step("Plot Generation", "info", "Generating single plot")
    
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
        st.error("No code provided to execute")
        return None
    
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
            st.error("No figure was created by the code")
            st.code(code)
            return None
        
        # Additional layout updates for subplots
        if is_subplot:
            fig.update_layout(
                height=600 if fig.layout.annotations and len(fig.layout.annotations) <= 4 else 800,
                showlegend=True,
                margin=dict(t=80, b=60, l=60, r=60)
            )
        
        return fig
        
    except Exception as e:
        st.error(f"Error executing Plotly code: {str(e)}")
        st.error("Code that failed:")
        st.code(code)
        return None


#-------------------------------------------------------------------------------------------------------------------------------------------------
# def generate_plotly_code(client, user_query, result_df, column_mapping):
#     """Step 5: Generate Plotly visualization code with proper column names"""
    
#     df_info = f"""
#     DataFrame shape: {result_df.shape}
#     Columns: {result_df.columns.tolist()}
#     Data types: {result_df.dtypes.to_dict()}
    
#     Column Details:
#     {json.dumps([{'name': col, 'type': str(result_df[col].dtype), 'sample_values': result_df[col].dropna().head(3).tolist()} for col in result_df.columns], indent=2, default=str)}
    
#     First few rows:
#     {result_df.head().to_string()}
#     """
    
#     prompt = f"""
#     Generate Python code using Plotly to visualize this data based on the user query:
    
#     User Query: "{user_query}"
    
#     Data Information:
#     {df_info}
    
#     Requirements:
#     1. Use plotly.express or plotly.graph_objects
#     2. The DataFrame variable name should be 'result_df'
#     3. Use the EXACT column names as shown in the Columns list above
#     4. Return only the Python code to create the figure between codestart and codeend markers
#     5. Assign the figure to a variable called 'fig'
#     6. Don't include fig.show() or any display commands
#     7. Choose the most appropriate chart type for the data and query
#     8. Include proper titles and labels
#     9. Handle any potential data type issues
    
#     Example format:
#     codestart
#     import plotly.express as px
#     fig = px.bar(result_df, x='column1', y='column2', title='Chart Title')
#     codeend
    
#     Your response must begin with 'codestart' and end with 'codeend'
#     """
    
#     response = call_groq_api(client, prompt)
#     if response:
#         # Extract code between codestart and codeend markers
#         match = re.search(r'codestart(.*?)codeend', response, re.DOTALL)
#         if match:
#             code = match.group(1).strip()
#             return code
#         else:
#             st.warning("Could not find code markers in the response. Trying to extract raw code...")
#             # Fallback to previous extraction method
#             code = re.sub(r'```python\n?|```\n?|Python Code:\s*', '', response).strip()
#             return code
#     return None

def generate_plotly_code(client, user_query, result_df, column_mapping):
    """Updated wrapper function for backward compatibility"""
    code, is_subplot = generate_enhanced_plotly_code(client, user_query, result_df, column_mapping)
    return code

# def execute_plotly_code(code, result_df):
#     """Step 6: Execute Plotly code and return figure"""
#     if not code:
#         st.error("No code provided to execute")
#         return None
    
#     try:
#         # Import required libraries in the execution environment
#         import numpy as np
#         import plotly.express as px
#         import plotly.graph_objects as go
#         import pandas as pd
        
#         # Create a safe execution environment
#         exec_globals = {
#             'px': px,
#             'go': go,
#             'result_df': result_df,
#             'pd': pd,
#             'np': np  # Now numpy is properly defined
#         }
        
#         # Execute the code
#         exec(code, exec_globals)
        
#         # Get the figure
#         fig = exec_globals.get('fig')
        
#         if fig is None:
#             st.error("No figure was created by the code")
#             st.code(code)  # Show the problematic code
#             return None
        
#         return fig
        
#     except Exception as e:
#         st.error(f"Error executing Plotly code: {str(e)}")
#         st.error("Code that failed:")
#         st.code(code)  # Display the code with syntax highlighting
#         return None
        
#     except Exception as e:
#         st.error(f"Error executing Plotly code: {str(e)}")
#         st.error("Code that failed:")
#         st.code(code)  # Display the code with syntax highlighting
#         return None


def execute_plotly_code(code, result_df):
    """Updated wrapper function for backward compatibility"""
    # Try to detect if it's subplot code
    is_subplot = 'make_subplots' in code if code else False
    return execute_enhanced_plotly_code(code, result_df, is_subplot)


def generate_analysis(client, user_query, result_df, chart_description, column_mapping):
    """Step 7: Generate analysis based on data and visualization with logging"""
    log_step("Analysis Generation", "started")
    
    try:
        # Prepare detailed data summary with column information
        column_details = []
        for col in result_df.columns:
            col_info = {
                'name': col,
                'type': str(result_df[col].dtype),
                'non_null_count': result_df[col].count(),
                'unique_values': result_df[col].nunique()
            }
            
            if result_df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'min': result_df[col].min(),
                    'max': result_df[col].max(),
                    'mean': result_df[col].mean()
                })
            
            column_details.append(col_info)
        
        df_summary = f"""
        Data Summary:
        - Shape: {result_df.shape}
        - Columns with Details: {json.dumps(column_details, indent=2, default=str)}
        
        Statistical Summary:
        {result_df.describe().to_string() if len(result_df.select_dtypes(include='number').columns) > 0 else "No numerical columns for statistics"}
        
        Sample Data (first 10 rows):
        {result_df.head(10).to_string()}
        
        Key-Value Pairs from Data:
        {json.dumps(result_df.head(5).to_dict('records'), indent=2, default=str)}
        """
        
        prompt = f"""
        Provide a comprehensive analysis of the data and visualization:
        
        Original User Query: "{user_query}"
        Chart Description: "{chart_description}"
        
        {df_summary}
        
        Please provide:
        1. Key insights from the data with specific values and column references
        2. Statistical findings with actual numbers from the dataset
        3. Direct answer to the original user query with supporting data
        4. Patterns or trends observed (mention specific column names and values)
        5. Recommendations or next steps based on the findings
        6. Any limitations or considerations
        
        IMPORTANT: 
        - Reference actual column names and values from the data
        - Include specific numbers and percentages where relevant
        - Maintain the key-value pair format when discussing data points
        - Be specific about what the data shows, not generic
        
        Keep the analysis clear, actionable, and relevant to the user's original question.
        """
        
        response = call_groq_api(client, prompt)
        
        if response:
            log_step("Analysis Generation", "success", f"Analysis generated (length: {len(response)})")
        else:
            log_step("Analysis Generation", "error", "No analysis generated")
        
        return response
        
    except Exception as e:
        log_step("Analysis Generation", "error", str(e))
        return None
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
# Main Streamlit App
def main():
    st.title("🤖 Groq-Powered Data Analysis Pipeline")
    st.markdown("Upload your data and ask questions - let AI handle the rest!")
    
    # Add logging display option
    with st.sidebar:
        st.header("Configuration")
        # api_key = st.text_input("Enter Groq API Key", type="password", help="Get your API key from console.groq.com")
        show_logs = st.checkbox("Show Detailed Logs", value=False)
        
    #     if api_key:
    #         env_path = '.env'
    #         load_dotenv(dotenv_path=env_path)
            
    #         groq_api_key = os.getenv("GROQ_API_KEY")
    #         os.environ["GROQ_API_KEY"] = groq_api_key
    
    # if not api_key:
    #     st.warning("Please enter your Groq API key in the sidebar to continue.")
    #     return
    
    # Initialize Groq client
    try:
        env_path = '.env'
        load_dotenv(dotenv_path=env_path)
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = groq_api_key
        client = Groq(api_key=groq_api_key)
        logger.info("Groq client initialized successfully")
    except Exception as e:
        st.error("Invalid API key. Please check your Groq API key.")
        logger.error(f"Failed to initialize Groq client: {str(e)}")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload the data file you want to analyze"
    )
    
    # User query input
    user_query = st.text_area(
        "What would you like to analyze?",
        placeholder="e.g., Show me the sales trends by month, Compare performance across regions, etc.",
        height=100
    )
    
    if uploaded_file and user_query and st.button("🚀 Start Analysis", type="primary"):
        
        # Create columns for progress tracking
        progress_col, status_col = st.columns([3, 1])
        
        with progress_col:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        # Create a container for logs if enabled
        if show_logs:
            log_container = st.expander("📋 Detailed Execution Logs", expanded=True)
        
        try:
            # Step 1: Understand query and identify tables
            status_text.text("🔍 Understanding your query...")
            progress_bar.progress(10)
            
            table_structure = understand_query_and_identify_tables(client, user_query)
            
            if table_structure:
                with st.expander("📋 Identified Requirements"):
                    st.json(table_structure)
            else:
                log_step("Analysis Pipeline", "warning", "Could not parse query structure, proceeding with default approach")
            
            # Step 2: Convert Excel to SQLite
            status_text.text("📊 Processing your data file...")
            progress_bar.progress(25)
            
            db_path, df, table_name, column_mapping = excel_to_sqlite(uploaded_file, table_structure)

            if db_path is None:
                log_step("Analysis Pipeline", "error", "Failed to create database from uploaded file")
                return
            
            if df is None or df.empty:
                log_step("Analysis Pipeline", "error", "Uploaded file is empty or could not be processed")
                return
            
            with st.expander("👀 Preview of your data"):
                # Show data with original column names
                display_df = df.copy()
                display_df.columns = [column_mapping.get(col, col) for col in display_df.columns]
                st.dataframe(display_df.head())
            
            if show_logs:
                with st.expander("🔧 Column Name Mapping"):
                    st.write("SQL Column Name → Original Column Name:")
                    for sql_col, orig_col in column_mapping.items():
                        st.write(f"`{sql_col}` → `{orig_col}`")
            
            # Step 3: Generate SQL query
            status_text.text("🔧 Generating SQL query...")
            progress_bar.progress(40)
            
            sql_query, actual_columns = generate_sql_query(client, user_query, table_name, db_path, df, column_mapping)
            
            if sql_query:
                # Validate and fix SQL query if needed
                sql_query = validate_and_fix_sql_query(client, sql_query, table_name, actual_columns, user_query, column_mapping)
                
                if sql_query:
                    with st.expander("📝 Generated SQL Query"):
                        st.code(sql_query, language='sql')
                else:
                    log_step("Analysis Pipeline", "error", "Failed to generate or validate SQL query")
                    return
            else:
                log_step("Analysis Pipeline", "error", "Failed to generate SQL query")
                return
            
            # Step 4: Execute SQL query
            status_text.text("⚡ Executing query...")
            progress_bar.progress(55)
            
            result_df, sql_error = execute_sql_query(db_path, sql_query)
            
            if sql_error:
                log_step("SQL Execution", "error", f"SQL Error: {sql_error}")
                st.info("💡 Try rephrasing your question or check if the column names match your data.")
                return
            
            if result_df is not None and not result_df.empty:
                # Restore original column names
                result_df = restore_original_column_names(result_df, column_mapping)
                
                with st.expander("📊 Query Results"):
                    st.dataframe(result_df)
            else:
                log_step("Analysis Pipeline", "warning", "Query executed but returned no results")
                st.warning("⚠️ Query executed but returned no results. Try a different query.")
                return
            
            # Step 5: Generate Plotly code
            # status_text.text("🎨 Creating visualization...")
            # progress_bar.progress(70)
            
            # plotly_code = generate_plotly_code(client, user_query, result_df, column_mapping)
            
            # if plotly_code:
            #     if show_logs:
            #         with st.expander("💻 Generated Plotly Code"):
            #             st.code(plotly_code, language='python')
            # else:
            #     log_step("Analysis Pipeline", "warning", "Could not generate visualization code")
            
            # # Step 6: Execute Plotly code
            # status_text.text("📈 Rendering chart...")
            # progress_bar.progress(85)
            
            # fig = None
            # if plotly_code:
            #     fig = execute_plotly_code(plotly_code, result_df)
            
            # if fig:
            #     st.plotly_chart(fig, use_container_width=True)
            # else:
            #     log_step("Visualization", "warning", "Could not create visualization, showing data table instead")
            #     st.info("📊 Visualization could not be created, but here's your data:")
            #     st.dataframe(result_df)


            status_text.text("🎨 Analyzing visualization needs...")
            progress_bar.progress(70)
            
            # Analyze data for potential subplots
            subplot_analysis = analyze_data_for_subplots(result_df)
            
            if subplot_analysis['needs_subplots']:
                log_step("Visualization Analysis", "info", 
                        f"Multiple subplots recommended: {', '.join(subplot_analysis['subplot_reasons'])}")
                
                # Show subplot analysis to user
                with st.expander("📊 Visualization Analysis"):
                    st.write("**Subplot Analysis Results:**")
                    st.write(f"- Reasons for subplots: {', '.join(subplot_analysis['subplot_reasons'])}")
                    st.write(f"- Suggested layout: {subplot_analysis['suggested_layout']}")
                    st.write(f"- Number of plot combinations: {len(subplot_analysis['plot_combinations'])}")
                    
                    if subplot_analysis['plot_combinations']:
                        st.write("**Plot Configurations:**")
                        for i, config in enumerate(subplot_analysis['plot_combinations']):
                            st.write(f"{i+1}. {config['type']}: {config}")
            
            # Generate enhanced plotly code (single or multiple plots)
            plotly_code, is_subplot = generate_enhanced_plotly_code(client, user_query, result_df, column_mapping)
            
            if plotly_code:
                if show_logs:
                    plot_type = "Subplot Code" if is_subplot else "Single Plot Code"
                    with st.expander(f"💻 Generated {plot_type}"):
                        st.code(plotly_code, language='python')
                        
                        # Show additional info for subplots
                        if is_subplot:
                            st.info("🎯 **Multiple subplots detected!** This visualization will show multiple related charts in one view.")
            else:
                log_step("Analysis Pipeline", "warning", "Could not generate visualization code")
            
            # Step 6: Execute enhanced plotly code
            status_text.text("📈 Rendering visualization...")
            progress_bar.progress(85)
            
            fig = None
            if plotly_code:
                fig = execute_enhanced_plotly_code(plotly_code, result_df, is_subplot)
                
                # Add subplot-specific information
                if fig and is_subplot:
                    st.success("✨ **Multi-panel Dashboard Created!** Each panel shows a different aspect of your data.")
            
            if fig:
                # Make the chart larger for subplots
                height = 700 if is_subplot else 500
                st.plotly_chart(fig, use_container_width=True, height=height)
                
                # Add interaction tips for subplots
                if is_subplot:
                    st.info("""
                    💡 **Subplot Interaction Tips:**
                    - Hover over each panel for detailed information
                    - Use the legend to show/hide data series
                    - Zoom and pan work on individual panels
                    - Double-click to reset zoom on all panels
                    """)
            else:
                log_step("Visualization", "warning", "Could not create visualization, showing data table instead")
                st.info("📊 Visualization could not be created, but here's your data:")
                st.dataframe(result_df)

            
            # Step 7: Generate analysis
            status_text.text("🧠 Generating insights...")
            progress_bar.progress(95)
            
            chart_description = f"Chart showing: {user_query}"
            analysis = generate_analysis(client, user_query, result_df, chart_description, column_mapping)
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            if analysis:
                st.markdown("## 📝 AI Analysis & Insights")
                st.markdown(analysis)
            else:
                log_step("Analysis Pipeline", "warning", "Could not generate detailed analysis")
                st.info("Analysis could not be generated, but your data query was successful!")
            
            # Cleanup
            if db_path and os.path.exists(db_path):
                os.unlink(db_path)
                logger.info("Temporary database cleaned up")
            
            log_step("Analysis Pipeline", "success", "Complete analysis pipeline finished successfully")
            
        except Exception as e:
            log_step("Analysis Pipeline", "error", f"Unexpected error in main pipeline: {str(e)}")
            progress_bar.progress(0)
            status_text.text("❌ Analysis failed")
            
            # Show error details if logs are enabled
            if show_logs:
                st.error(f"Detailed error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Powered by Groq AI** | Upload your data, ask questions, and get instant insights with visualizations!"
    )
    
    # Add troubleshooting section
    with st.expander("🔧 Troubleshooting Tips"):
        st.markdown("""
        **Common Issues and Solutions:**
        
        1. **"Query must be a string" Error:**
           - This usually means the AI couldn't generate a proper SQL query
           - Try rephrasing your question more clearly
           - Make sure your data has clear column names
        
        2. **No Results Returned:**
           - Check if your question matches the data available
           - Try asking for basic statistics first (e.g., "show me all data")
        
        3. **Visualization Errors:**
           - The system will show your data in a table if charts fail
           - Try asking for different types of analysis
        
        4. **Column Name Issues:**
           - Enable "Show Detailed Logs" to see column name mappings
           - Special characters in column names are automatically cleaned
        
        **Tips for Better Results:**
        - Use clear, specific questions
        - Mention column names that exist in your data
        - Ask for one thing at a time initially
        """)

if __name__ == "__main__":
    main()