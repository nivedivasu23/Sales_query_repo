import json
from Config import call_groq_api
import re


def understand_query_and_identify_tables(client, user_query):
    """Step 1: Understand query and identify required tables with enhanced logging"""
    #log_step("Query Understanding", "started")
    
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
                    #log_step("Query Understanding", "success", f"Identified {len(parsed_json.get('required_tables', []))} required tables")
                    return parsed_json
                else:
                    print("Query Understanding", "warning", "No JSON found in response")
                    #log_step("Query Understanding", "warning", "No JSON found in response")
            except json.JSONDecodeError as e:
                print("Query Understanding", "error", f"JSON parsing failed: {str(e)}")
                #log_step("Query Understanding", "error", f"JSON parsing failed: {str(e)}")
        else:
            print("Query Understanding", "error", "No response from API")
            #log_step("Query Understanding", "error", "No response from API")
            
    except Exception as e:
        print("Query Understanding", "error", str(e))
        #log_step("Query Understanding", "error", str(e))
    
    return None