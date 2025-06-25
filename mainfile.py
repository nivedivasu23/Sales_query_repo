import logging
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
from AnalysisData import understand_query_and_identify_tables
from dataProcess import excel_to_sqlite
from SQLgenerator import generate_sql_query ,validate_and_fix_sql_query,execute_sql_query,restore_original_column_names
from Plotlyplot import analyze_data_for_subplots,generate_enhanced_plotly_code,execute_enhanced_plotly_code
from insights import generate_analysis




logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
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
    # try:
        env_path = '.env'
        load_dotenv(dotenv_path=env_path)
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = groq_api_key
        client = Groq(api_key=groq_api_key)
    #     logger.info("Groq client initialized successfully")
    # except Exception as e:
    #     st.error("Invalid API key. Please check your Groq API key.")
    #     logger.error(f"Failed to initialize Groq client: {str(e)}")
    #     return
    
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