import re
from dataProcess import inspect_table_schema
from Config import call_groq_api
import json
import sqlite3
import pandas as pd


def generate_sql_query(client, user_query, table_name, db_path, df_info, column_mapping):
    """Step 3: Generate SQL query based on user query and actual table structure with enhanced logging"""
    print("SQL Query Generation", "started")
    
    try:
        # Get actual table schema from database
        actual_columns, sample_data = inspect_table_schema(db_path, table_name)
        
        if not actual_columns:
            print("SQL Query Generation", "error", "Could not retrieve table schema")
            return None, None
        
        # Get column names and types
        column_names = [col['name'] for col in actual_columns]
        column_details = f"Available columns with types: {actual_columns}"
        
        print(f"Available SQL columns: {column_names}")
        
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
                print("SQL Query Generation", "error", "Generated query is empty")
                return None, None
            
            # Basic validation that it looks like SQL
            if not any(keyword in sql_query.upper() for keyword in ['SELECT', 'FROM']):
                print("SQL Query Generation", "error", f"Generated text doesn't look like SQL: {sql_query[:100]}")
                return None, None
            
            print(f"Generated SQL query: {sql_query}")
            print("SQL Query Generation", "success", f"Query length: {len(sql_query)} characters")
            return sql_query, column_names
        else:
            print("SQL Query Generation", "error", "No response from API")
            return None, None
            
    except Exception as e:
        print("SQL Query Generation", "error", str(e))
        return None, None

def validate_and_fix_sql_query(client, sql_query, table_name, actual_columns, user_query, column_mapping):
    """Validate SQL query against actual columns and fix if needed with logging"""
    print("SQL Query Validation", "started")
    
    try:
        # Ensure sql_query is a string
        if not isinstance(sql_query, str):
            print("SQL Query Validation", "error", f"SQL query is not a string: {type(sql_query)}")
            return None
        
        if not sql_query.strip():
            print("SQL Query Validation", "error", "SQL query is empty")
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
            print("SQL Query Validation", "warning", f"Missing columns detected: {missing_columns}")
            
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
                print("SQL Query Validation", "success", "Query automatically fixed")
                return fixed_query
        
        print("SQL Query Validation", "success", "Query validation passed")
        return sql_query
        
    except Exception as e:
        print("SQL Query Validation", "error", str(e))
        return sql_query

def execute_sql_query(db_path, sql_query):
    """Step 4: Execute SQL query and return results with enhanced logging"""
    print("SQL Query Execution", "started")
    
    try:
        # Validate inputs
        if not isinstance(sql_query, str):
            error_msg = f"SQL query must be a string, got {type(sql_query)}: {sql_query}"
            print("SQL Query Execution", "error", error_msg)
            return None, error_msg
        
        if not sql_query.strip():
            error_msg = "SQL query is empty or whitespace"
            print("SQL Query Execution", "error", error_msg)
            return None, error_msg
        
        print(f"Executing SQL query: {sql_query}")
        
        conn = sqlite3.connect(db_path)
        result_df = pd.read_sql_query(sql_query, conn)
        conn.close()
        
        print(f"Query executed successfully. Result shape: {result_df.shape}")
        print("SQL Query Execution", "success", f"Retrieved {len(result_df)} rows, {len(result_df.columns)} columns")
        
        return result_df, None
        
    except Exception as e:
        error_msg = str(e)
        print("SQL Query Execution", "error", error_msg)
        return None, error_msg

def restore_original_column_names(result_df, column_mapping):
    """Restore original column names in the result DataFrame with logging"""
    print("Column Name Restoration", "started")
    
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
        print(f"Restored column names: {new_columns}")
        print("Column Name Restoration", "success", f"Restored {len(new_columns)} column names")
        
        return result_df
        
    except Exception as e:
        print("Column Name Restoration", "error", str(e))
        return result_df
