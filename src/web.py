"""Streamlit web interface for BankingLLM system."""

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
        page_title="BankingLLM Data Analyst",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load custom CSS
    css_path = Path(__file__).parent.parent / "static" / "css" / "custom.css"
    with open(css_path) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Animated header
    st.markdown(
        """
        <div class="main-header">
            <h1>🏦 BankingLLM Data Analyst</h1>
            <p>Transform natural language queries into SQL and get professional Excel reports instantly!</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_sidebar():
    """Display sidebar with sample queries and statistics."""
    with st.sidebar:
        # Animated sidebar header
        st.markdown(
            """
            <div class="sidebar-header">
                <h3>📊 Database Info</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Database statistics with animated cards
        try:
            stats = db_manager.get_database_stats()

            # Animated stats cards
            st.markdown(
                f"""
                <div class="stats-card">
                    <h4>👥 Clients</h4>
                    <p style="font-size: 1.5rem; font-weight: bold; color: var(--primary);">{stats['clients']:,}</p>
                </div>
                <div class="stats-card">
                    <h4>🏦 Accounts</h4>
                    <p style="font-size: 1.5rem; font-weight: bold; color: var(--secondary);">{stats['accounts']:,}</p>
                </div>
                <div class="stats-card">
                    <h4>💸 Transactions</h4>
                    <p style="font-size: 1.5rem; font-weight: bold; color: var(--accent);">{stats['transactions']:,}</p>
                </div>
                <div class="stats-card">
                    <h4>🌍 Regions</h4>
                    <p style="font-size: 1.5rem; font-weight: bold; color: var(--warning);">{len(stats['regions'])}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            with st.expander("📍 Regions"):
                for region in stats['regions']:
                    st.write(f"• {region}")

        except Exception as e:
            st.error("Database not initialized")
            if st.button("Initialize Database"):
                setup_database()

        # Sample queries with better styling
        st.markdown("### 💡 Sample Queries")
        samples = llm_service.suggest_sample_queries()

        for i, sample in enumerate(samples[:5]):  # Show first 5
            if st.button(f"📝 {sample[:40]}...", key=f"sample_{i}"):
                st.session_state.selected_sample = sample

        # Query history with better styling
        if st.session_state.query_history:
            st.markdown("### 📜 Recent Queries")
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
    # Animated query container
    st.markdown(
        """
        <div class="query-container">
            <h3>🔍 Natural Language Query Interface</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Query input with better styling
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
            height=120,
            placeholder="e.g., Show total transactions by region for 2024",
            key="query_input"
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

    # Custom loading animation for SQL generation
    with st.container():
        st.markdown(
            """
            <div class="loading-container">
                <div class="loading-dots">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
                <p style="margin-top: 1rem; color: var(--primary); font-weight: 500;">🤖 Generating SQL query...</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Generate SQL
        llm_result = llm_service.generate_sql(user_query)

        if not llm_result['success']:
            st.error(f"❌ Error generating SQL: {llm_result.get('error', 'Unknown error')}")
            return

    # Display generated SQL with better styling
    st.markdown("### 🔧 Generated SQL Query")
    st.code(llm_result['sql_query'], language='sql')

    # Custom loading animation for query execution
    with st.container():
        st.markdown(
            """
            <div class="loading-container">
                <div class="loading-dots">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
                <p style="margin-top: 1rem; color: var(--secondary); font-weight: 500;">⚡ Executing query...</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        try:
            results = db_manager.execute_query(llm_result['sql_query'])

            if not results:
                st.warning("Query executed successfully but returned no results.")
                return

            # Animated success message
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, var(--secondary) 0%, #48bb78 100%);
                            color: white; padding: 1rem; border-radius: 0.5rem;
                            margin: 1rem 0; text-align: center; font-weight: 500;
                            animation: fadeInScale 0.5s ease-out;">
                    ✅ Query returned {len(results)} rows
                </div>
                """,
                unsafe_allow_html=True
            )

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

    # Results tabs with better styling
    st.markdown("### 📊 Query Results")
    
    tab1, tab2, tab3 = st.tabs(["📊 Data Table", "📈 Visualizations", "📋 Summary"])

    with tab1:
        # Styled dataframe
        st.dataframe(df, use_container_width=True)

        # Enhanced download buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv_data,
                file_name=f"bank_analysis_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            if st.button("📊 Generate Excel Report", use_container_width=True):
                generate_excel_report(results, query_info)

        with col3:
            # Display query explanation
            if st.button("💬 Explain Query", use_container_width=True):
                display_query_explanation(query_info['sql_query'])

    with tab2:
        display_charts(df)

    with tab3:
        display_summary(df)


def generate_excel_report(results: list, query_info: dict):
    """Generate and provide Excel report download."""
    try:
        # Custom loading animation for Excel generation
        with st.container():
            st.markdown(
                """
                <div class="loading-container">
                    <div class="loading-dots">
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                        <div class="loading-dot"></div>
                    </div>
                    <p style="margin-top: 1rem; color: var(--accent); font-weight: 500;">📊 Generating Excel report...</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            excel_path = excel_exporter.export_query_results(results, query_info)

            # Read the file for download
            with open(excel_path, 'rb') as f:
                excel_data = f.read()

            # Animated success message
            st.markdown(
                """
                <div style="background: linear-gradient(135deg, var(--accent) 0%, #f6e05e 100%);
                            color: white; padding: 1rem; border-radius: 0.5rem;
                            margin: 1rem 0; text-align: center; font-weight: 500;
                            animation: fadeInScale 0.5s ease-out;">
                    ✅ Excel report generated successfully!
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.download_button(
                "📥 Download Excel Report",
                excel_data,
                file_name=os.path.basename(excel_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    except Exception as e:
        st.error(f"❌ Error generating Excel report: {e}")


def display_charts(df: pd.DataFrame):
    """Display interactive charts based on data."""
    if df.empty:
        st.info("No data to display")
        return

    st.markdown("### 📈 Data Visualizations")

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

    if len(numeric_cols) == 0:
        st.info("No numeric data for charts")
        return

    # Chart selection with better styling
    chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Line Chart", "Area Chart"])

    if categorical_cols and numeric_cols:
        col1, col2 = st.columns(2)

        with col1:
            x_axis = st.selectbox("X-axis", categorical_cols + numeric_cols)
        with col2:
            y_axis = st.selectbox("Y-axis", numeric_cols)

        # Create chart based on selection with animation
        chart_container = st.empty()
        
        if chart_type == "Bar Chart":
            chart_container.bar_chart(data=df.set_index(x_axis)[y_axis])
        elif chart_type == "Line Chart":
            chart_container.line_chart(data=df.set_index(x_axis)[y_axis])
        elif chart_type == "Area Chart":
            chart_container.area_chart(data=df.set_index(x_axis)[y_axis])
    else:
        # Simple numeric display
        if len(numeric_cols) > 0:
            st.line_chart(df[numeric_cols])


def display_summary(df: pd.DataFrame):
    """Display data summary statistics."""
    if df.empty:
        st.info("No data to summarize")
        return

    st.markdown("### 📋 Data Summary")

    # Basic info with animated metric cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    # Column statistics with better styling
    st.markdown("#### Column Statistics")

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
    with st.container():
        st.markdown(
            """
            <div class="loading-container">
                <div class="loading-dots">
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                    <div class="loading-dot"></div>
                </div>
                <p style="margin-top: 1rem; color: var(--primary-light); font-weight: 500;">🧠 Generating explanation...</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        explanation = llm_service.get_query_explanation(sql_query)
        st.info(f"💡 **Query Explanation**: {explanation}")


def main():
    """Main Streamlit application."""
    init_session_state()
    display_header()
    display_sidebar()

    # Main content area - handled in run_query_interface
    run_query_interface()

    # Footer with better styling
    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
            <p>BankingLLM Data Analyst v1.0 | Built with Streamlit, FastAPI, and Ollama</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()