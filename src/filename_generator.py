"""Fast filename generator for Excel exports without AI requests."""

import re
from datetime import datetime
from typing import Dict, List, Optional


class FastFilenameGenerator:
    """Generate descriptive filenames for Excel exports by analyzing SQL queries and results."""

    def __init__(self):
        # Define common banking terms for better filename generation
        self.banking_terms = {
            'transactions': 'Transactions',
            'accounts': 'Accounts',
            'clients': 'Clients',
            'customers': 'Clients',
            'regions': 'Regions',
            'branches': 'Branches',
            'amount': 'Amount',
            'balance': 'Balance',
            'date': 'Date',
            'month': 'Month',
            'year': 'Year',
            'quarter': 'Quarter',
            'status': 'Status',
            'type': 'Type'
        }

    def generate_filename(self, sql_query: str, results: List[Dict]) -> str:
        """
        Generate a descriptive filename based on SQL query and results.
        
        Args:
            sql_query: The SQL query that generated the results
            results: The query results as a list of dictionaries
            
        Returns:
            A descriptive filename for the Excel export
        """
        # Extract key information from SQL
        tables = self._extract_tables(sql_query)
        operations = self._extract_operations(sql_query)
        time_period = self._extract_time_period(sql_query)
        group_by = self._extract_group_by(sql_query)
        
        # Analyze results
        primary_column = self._identify_primary_column(results)
        
        # Generate filename based on patterns
        filename = self._apply_filename_template(
            tables, operations, time_period, group_by, primary_column
        )
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name, extension = filename.rsplit('.', 1)
        filename = f"{base_name}_{timestamp}.{extension}"
        
        return filename

    def _extract_tables(self, sql_query: str) -> List[str]:
        """Extract table names from SQL query."""
        tables = []
        
        # Find table names after FROM
        from_pattern = r'FROM\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        tables.extend(re.findall(from_pattern, sql_query, re.IGNORECASE))
        
        # Find table names after JOIN
        join_patterns = [
            r'JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'INNER JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'LEFT JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'RIGHT JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        ]
        
        for pattern in join_patterns:
            tables.extend(re.findall(pattern, sql_query, re.IGNORECASE))
        
        # Convert to banking terms and remove duplicates
        processed_tables = []
        for table in tables:
            table_lower = table.lower()
            if table_lower in self.banking_terms:
                processed_tables.append(self.banking_terms[table_lower])
            else:
                # Convert snake_case to CamelCase
                processed_tables.append(''.join(word.title() for word in table.split('_')))
        
        return list(set(processed_tables))

    def _extract_operations(self, sql_query: str) -> List[str]:
        """Extract aggregate operations from SQL query."""
        operations = []
        
        # Look for aggregate functions
        operation_patterns = {
            'COUNT': 'Count',
            'SUM': 'Sum',
            'AVG': 'Average',
            'MAX': 'Maximum',
            'MIN': 'Minimum',
            'DISTINCT': 'Unique'
        }
        
        for op_sql, op_name in operation_patterns.items():
            if f'{op_sql}(' in sql_query.upper():
                operations.append(op_name)
        
        return operations

    def _extract_time_period(self, sql_query: str) -> Optional[str]:
        """Extract time period information from SQL query."""
        # Look for time-based functions
        time_patterns = {
            r'YEAR\s*\(\s*([^)]+)\s*\)': 'Year',
            r'MONTH\s*\(\s*([^)]+)\s*\)': 'Month',
            r'DAY\s*\(\s*([^)]+)\s*\)': 'Day',
            r'QUARTER\s*\(\s*([^)]+)\s*\)': 'Quarter',
            r'DATE_TRUNC\s*\(\s*[\'"]([^\'"]+)[\'"]\s*,': lambda m: m.group(1).title()
        }
        
        for pattern, replacement in time_patterns.items():
            match = re.search(pattern, sql_query, re.IGNORECASE)
            if match:
                if callable(replacement):
                    return replacement(match)
                return replacement
        
        # Look for WHERE clauses with date conditions
        date_conditions = [
            r'WHERE\s+[^=]+=\s*[\'"](\d{4})[\'"]',  # Year filter
            r'WHERE\s+[^=]+=\s*[\'"](\d{4}-\d{2})[\'"]',  # Year-month filter
        ]
        
        for pattern in date_conditions:
            match = re.search(pattern, sql_query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None

    def _extract_group_by(self, sql_query: str) -> List[str]:
        """Extract GROUP BY columns from SQL query."""
        group_by_pattern = r'GROUP\s+BY\s+(.+?)(?:\s+ORDER\s+BY|\s+HAVING|\s+LIMIT|$)'
        match = re.search(group_by_pattern, sql_query, re.IGNORECASE | re.DOTALL)
        
        if not match:
            return []
        
        group_by_clause = match.group(1)
        columns = [col.strip() for col in group_by_clause.split(',')]
        
        # Process column names
        processed_columns = []
        for col in columns:
            # Remove table prefixes and function calls
            col = re.sub(r'^[a-zA-Z_][a-zA-Z0-9_]*\.', '', col)
            col = re.sub(r'\([^)]+\)', '', col)
            col = col.strip()
            
            # Convert to banking terms or CamelCase
            col_lower = col.lower()
            if col_lower in self.banking_terms:
                processed_columns.append(self.banking_terms[col_lower])
            else:
                processed_columns.append(''.join(word.title() for word in col.split('_')))
        
        return processed_columns

    def _identify_primary_column(self, results: List[Dict]) -> Optional[str]:
        """Identify the primary column from results."""
        if not results:
            return None
        
        # Get column names from first result
        columns = list(results[0].keys())
        
        # Prioritize certain column types
        priority_columns = ['name', 'title', 'type', 'status', 'category', 'region']
        
        for col in columns:
            if col.lower() in priority_columns:
                return col.title()
        
        # If no priority column found, return the first column
        if columns:
            return columns[0].title()
        
        return None

    def _apply_filename_template(self, tables: List[str], operations: List[str], 
                               time_period: Optional[str], group_by: List[str], 
                               primary_column: Optional[str]) -> str:
        """Apply filename template based on extracted information."""
        # Build filename components
        table_part = tables[0] if tables else 'Data'
        
        operation_part = ''
        if operations:
            operation_part = '_' + operations[0]
        
        group_part = ''
        if group_by:
            group_part = 'By' + group_by[0]
        
        time_part = ''
        if time_period:
            time_part = '_' + str(time_period)
        
        # Special case for single table with no operations
        if not operations and not group_by and not time_period:
            if primary_column and primary_column.lower() != table_part.lower():
                group_part = 'By' + primary_column
        
        # Combine parts
        filename = f"{table_part}{operation_part}{group_part}{time_part}.xlsx"
        
        # Clean up filename
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        filename = re.sub(r'_+', '_', filename)  # Remove multiple underscores
        filename = filename.strip('_')  # Remove leading/trailing underscores
        
        return filename


# Global filename generator instance
filename_generator = FastFilenameGenerator()