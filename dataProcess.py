import pandas as pd
import re
import tempfile
import sqlite3



def clean_column_names(df):
    """Clean column names to ensure SQL compatibility and handle duplicates with logging"""

    print("Column Name Cleaning", "started")
    
    try:
        # Store original column names mapping
        original_columns = df.columns.tolist()
        print(f"Original columns: {original_columns}")
        
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
        
        print(f"Cleaned columns: {clean_columns}")
        print(f"Column mapping created: {len(column_mapping)} mappings")
        
        print("Column Name Cleaning", "success", f"Processed {len(original_columns)} columns")
        return df, column_mapping
        
    except Exception as e:
        print("Column Name Cleaning", "error", str(e))
        return df, {}

def excel_to_sqlite(uploaded_file, table_structure):
    """Step 2: Convert Excel to SQLite database with improved column handling and logging"""
    print("Excel to SQLite Conversion", "started")
    
    try:
        # Read Excel file
        print(f"Reading file: {uploaded_file.name}")
        if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        
        print(f"File loaded successfully. Shape: {df.shape}")
        
        if df.empty:
            print("Excel to SQLite Conversion", "error", "DataFrame is empty")
            return None, None, None, None
        
        # Clean column names for SQL compatibility
        df, column_mapping = clean_column_names(df)
        
        # Create temporary SQLite database
        temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db_path = temp_db.name
        temp_db.close()  # Close the file handle so SQLite can open it
        
        print(f"Created temporary database: {temp_db_path}")
        
        conn = sqlite3.connect(temp_db_path)
        
        # If we have table structure, use the first table name, otherwise use 'data'
        table_name = 'data'
        if table_structure and 'required_tables' in table_structure:
            table_name = table_structure['required_tables'][0]['table_name']
        
        print(f"Using table name: {table_name}")
        
        # Store data in SQLite
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Data stored in SQLite table '{table_name}'")
        
        conn.close()
        
        print("Excel to SQLite Conversion", "success", f"Created table '{table_name}' with {len(df)} rows")
        return temp_db_path, df, table_name, column_mapping
        
    except Exception as e:
        print("Excel to SQLite Conversion", "error", str(e))
        return None, None, None, None

def inspect_table_schema(db_path, table_name):
    """Get actual table schema from SQLite database with logging"""
    print("Schema Inspection", "started")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table schema
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        print(f"Found {len(columns_info)} columns in table")
        
        # Get sample data
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
        sample_data = cursor.fetchall()
        print(f"Retrieved {len(sample_data)} sample rows")
        
        conn.close()
        
        # Format column information
        columns = [{'name': col[1], 'type': col[2]} for col in columns_info]
        print(f"Column details: {columns}")
        
        print("Schema Inspection", "success", f"Inspected {len(columns)} columns")
        return columns, sample_data
        
    except Exception as e:
        print("Schema Inspection", "error", str(e))
        return None, None