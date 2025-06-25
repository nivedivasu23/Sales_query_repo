
import json
from Config import call_groq_api

def generate_analysis(client, user_query, result_df, chart_description, column_mapping):
    """Step 7: Generate analysis based on data and visualization with logging"""
    print("Analysis Generation", "started")
    
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
            print("Analysis Generation", "success", f"Analysis generated (length: {len(response)})")
        else:
            print("Analysis Generation", "error", "No analysis generated")
        
        return response
        
    except Exception as e:
        print("Analysis Generation", "error", str(e))
        return None