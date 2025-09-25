"""Streamlit web interface for Bank AI LLM system."""

import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger

from .config import settings
from .database import db_manager
from .excel_export import excel_exporter
from .llm_service import llm_service


def init_session_state():
    """Initialize Streamlit session state."""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None


def display_header():
    """Display application header."""
    st.set_page_config(
        page_title="Bank AI LLM Data Analyst",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🏦 Bank AI LLM Data Analyst")
    st.markdown(
        """
        <div style='text-align: center; color: #666; margin-bottom: 2em;'>
        Transform natural language queries into SQL and get professional Excel reports instantly!
        </div>
        """,
        unsafe_allow_html=True
    )


def display_sidebar():
    """Display sidebar with sample queries and statistics."""
    with st.sidebar:
        st.header("📊 Database Info")

        # Database statistics
        try:
            stats = db_manager.get_database_stats()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Clients", f"{stats['clients']:,}")
                st.metric("Accounts", f"{stats['accounts']:,}")
            with col2:
                st.metric("Transactions", f"{stats['transactions']:,}")
                st.metric("Regions", len(stats['regions']))

            with st.expander("Regions"):
                for region in stats['regions']:
                    st.write(f"• {region}")

        except Exception as e:
            st.error("Database not initialized")
            if st.button("Initialize Database"):
                setup_database()

        # Sample queries
        st.header("💡 Sample Queries")
        samples = llm_service.suggest_sample_queries()

        for i, sample in enumerate(samples[:5]):  # Show first 5
            if st.button(f"📝 {sample[:40]}...", key=f"sample_{i}"):
                st.session_state.selected_sample = sample

        # Query history
        if st.session_state.query_history:
            st.header("📜 Recent Queries")
            for i, query in enumerate(reversed(st.session_state.query_history[-5:])):
                if st.button(f"🔄 {query[:30]}...", key=f"history_{i}"):
                    st.session_state.selected_history = query


def setup_database():
    """Setup database with progress indicator."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Creating database tables...")
        progress_bar.progress(25)
        db_manager.create_tables()

        status_text.text("Generating mock data (this may take a few minutes)...")
        progress_bar.progress(50)
        db_manager.generate_mock_data()

        progress_bar.progress(100)
        status_text.text("Database setup completed!")
        st.success("✅ Database initialized successfully!")
        time.sleep(2)
        st.experimental_rerun()

    except Exception as e:
        st.error(f"Error setting up database: {e}")


def run_query_interface():
    """Main query interface."""
    # Query input
    col1, col2 = st.columns([4, 1])

    with col1:
        # Check for selected sample or history
        default_query = ""
        if hasattr(st.session_state, 'selected_sample'):
            default_query = st.session_state.selected_sample
            del st.session_state.selected_sample
        elif hasattr(st.session_state, 'selected_history'):
            default_query = st.session_state.selected_history
            del st.session_state.selected_history

        user_query = st.text_area(
            "Enter your natural language query:",
            value=default_query,
            height=100,
            placeholder="e.g., Show total transactions by region for 2024"
        )

    with col2:
        st.write("")  # Spacing
        run_query = st.button("🔍 Run Query", type="primary", use_container_width=True)
        export_format = st.selectbox("Export Format", ["Excel", "CSV"])

    # Execute query
    if run_query and user_query.strip():
        execute_query(user_query, export_format.lower())


def execute_query(user_query: str, export_format: str = "excel"):
    """Execute the query and display results."""
    # Add to query history
    if user_query not in st.session_state.query_history:
        st.session_state.query_history.append(user_query)

    with st.spinner("🤖 Generating SQL query..."):
        # Generate SQL
        llm_result = llm_service.generate_sql(user_query)

        if not llm_result['success']:
            st.error(f"❌ Error generating SQL: {llm_result.get('error', 'Unknown error')}")
            return

    # Display generated SQL
    st.subheader("🔧 Generated SQL Query")
    st.code(llm_result['sql_query'], language='sql')

    # Execute query
    with st.spinner("⚡ Executing query..."):
        try:
            results = db_manager.execute_query(llm_result['sql_query'])

            if not results:
                st.warning("Query executed successfully but returned no results.")
                return

            st.success(f"✅ Query returned {len(results)} rows")

            # Store results in session state
            st.session_state.current_results = {
                'data': results,
                'query_info': llm_result
            }

        except Exception as e:
            st.error(f"❌ Error executing query: {e}")
            return

    # Display results
    display_results(results, llm_result, export_format)


def display_results(results: list, query_info: dict, export_format: str):
    """Display query results and export options."""
    # Convert to DataFrame for better display
    df = pd.DataFrame(results)

    # Results tabs
    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Charts", "📋 Summary"])

    with tab1:
        st.subheader("Query Results")
        st.dataframe(df, use_container_width=True)

        # Download buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv_data,
                file_name=f"bank_analysis_{int(time.time())}.csv",
                mime="text/csv"
            )

        with col2:
            if st.button("📊 Generate Excel Report"):
                generate_excel_report(results, query_info)

        with col3:
            # Display query explanation
            if st.button("💬 Explain Query"):
                display_query_explanation(query_info['sql_query'])

    with tab2:
        display_charts(df)

    with tab3:
        display_summary(df)


def generate_excel_report(results: list, query_info: dict):
    """Generate and provide Excel report download."""
    try:
        with st.spinner("📊 Generating Excel report..."):
            excel_path = excel_exporter.export_query_results(results, query_info)

            # Read the file for download
            with open(excel_path, 'rb') as f:
                excel_data = f.read()

            st.success("✅ Excel report generated!")
            st.download_button(
                "📥 Download Excel Report",
                excel_data,
                file_name=os.path.basename(excel_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"❌ Error generating Excel report: {e}")


def display_charts(df: pd.DataFrame):
    """Display interactive charts based on data."""
    if df.empty:
        st.info("No data to display")
        return

    st.subheader("📈 Data Visualizations")

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    if len(numeric_cols) == 0:
        st.info("No numeric data for charts")
        return

    # Chart selection
    chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Area Chart"])

    if categorical_cols and numeric_cols:
        col1, col2 = st.columns(2)

        with col1:
            x_axis = st.selectbox("X-axis", categorical_cols + numeric_cols)
        with col2:
            y_axis = st.selectbox("Y-axis", numeric_cols)

        # Create chart based on selection
        if chart_type == "Bar Chart":
            st.bar_chart(data=df.set_index(x_axis)[y_axis])
        elif chart_type == "Line Chart":
            st.line_chart(data=df.set_index(x_axis)[y_axis])
        elif chart_type == "Area Chart":
            st.area_chart(data=df.set_index(x_axis)[y_axis])
    else:
        # Simple numeric display
        if len(numeric_cols) > 0:
            st.line_chart(df[numeric_cols])


def display_summary(df: pd.DataFrame):
    """Display data summary statistics."""
    if df.empty:
        st.info("No data to summarize")
        return

    st.subheader("📋 Data Summary")

    # Basic info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    # Column statistics
    st.subheader("Column Statistics")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numeric_cols:
        st.write("**Numeric Columns:**")
        st.dataframe(df[numeric_cols].describe())

    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
    if categorical_cols:
        st.write("**Categorical Columns:**")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            st.write(f"• **{col}**: {unique_count} unique values")
            if unique_count <= 10:  # Show values if not too many
                st.write(f"  Values: {', '.join(map(str, df[col].unique()))}")


def display_query_explanation(sql_query: str):
    """Display query explanation."""
    with st.spinner("🧠 Generating explanation..."):
        explanation = llm_service.get_query_explanation(sql_query)
        st.info(f"💡 **Query Explanation**: {explanation}")


def main():
    """Main Streamlit application."""
    init_session_state()
    display_header()
    display_sidebar()

    # Main content area
    st.header("🔍 Natural Language Query Interface")
    run_query_interface()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888; font-size: 0.8em;'>
        Bank AI LLM Data Analyst Assistant v1.0 | Built with Streamlit, FastAPI, and Ollama
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()