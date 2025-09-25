"""Excel export engine with automatic chart generation."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, PieChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows

from .config import settings


class ExcelExporter:
    """Professional Excel export with automatic chart generation."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or settings.excel_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Professional styling colors (Banking theme)
        self.colors = {
            'primary': '1f4e79',      # Dark blue
            'secondary': '5b9bd5',     # Light blue
            'accent': '70ad47',        # Green
            'warning': 'ffc000',       # Orange
            'header': 'f2f2f2',       # Light gray
            'text': '404040'           # Dark gray
        }

    def export_query_results(
        self,
        data: List[Dict],
        query_info: Dict,
        filename: Optional[str] = None
    ) -> str:
        """Export query results to Excel with professional formatting and charts."""

        if not data:
            raise ValueError("No data to export")

        # Create filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bank_analysis_{timestamp}.xlsx"

        filepath = self.output_dir / filename
        logger.info(f"Exporting data to Excel: {filepath}")

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)

        # Create workbook and worksheets
        wb = Workbook()
        ws_data = wb.active
        ws_data.title = "Data"
        ws_summary = wb.create_sheet("Summary")

        # Export main data
        self._write_data_sheet(ws_data, df, query_info)

        # Create summary and charts
        self._create_summary_sheet(ws_summary, df, query_info)

        # Add charts based on data type
        chart_sheet = wb.create_sheet("Charts")
        self._add_charts(wb, chart_sheet, df)

        # Save workbook
        wb.save(filepath)
        logger.info(f"Excel export completed: {filepath}")

        return str(filepath)

    def _write_data_sheet(self, ws, df: pd.DataFrame, query_info: Dict):
        """Write data to the main data sheet with professional formatting."""

        # Add header with query information
        ws.merge_cells('A1:E1')
        ws['A1'] = "Bank Data Analysis Report"
        ws['A1'].font = Font(size=16, bold=True, color=self.colors['primary'])
        ws['A1'].alignment = Alignment(horizontal='center')

        # Add query info
        ws['A3'] = "Query:"
        ws['A3'].font = Font(bold=True, color=self.colors['text'])
        ws['B3'] = query_info.get('user_query', 'N/A')

        ws['A4'] = "Generated SQL:"
        ws['A4'].font = Font(bold=True, color=self.colors['text'])
        ws['B4'] = query_info.get('sql_query', 'N/A')

        ws['A5'] = "Export Time:"
        ws['A5'].font = Font(bold=True, color=self.colors['text'])
        ws['B5'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Add data starting from row 7
        start_row = 7

        # Write headers
        for col_idx, column in enumerate(df.columns, 1):
            cell = ws.cell(row=start_row, column=col_idx, value=column)
            cell.font = Font(bold=True, color='FFFFFF')
            cell.fill = PatternFill(start_color=self.colors['primary'], end_color=self.colors['primary'], fill_type='solid')
            cell.alignment = Alignment(horizontal='center')

        # Write data
        for row_idx, row in enumerate(df.itertuples(index=False), start_row + 1):
            for col_idx, value in enumerate(row, 1):
                cell = ws.cell(row=row_idx, column=col_idx, value=value)
                if row_idx % 2 == 0:  # Alternate row coloring
                    cell.fill = PatternFill(start_color=self.colors['header'], end_color=self.colors['header'], fill_type='solid')

        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width

    def _create_summary_sheet(self, ws, df: pd.DataFrame, query_info: Dict):
        """Create a summary sheet with key statistics."""

        # Title
        ws.merge_cells('A1:D1')
        ws['A1'] = "Data Summary"
        ws['A1'].font = Font(size=14, bold=True, color=self.colors['primary'])
        ws['A1'].alignment = Alignment(horizontal='center')

        # Basic statistics
        row = 3
        ws[f'A{row}'] = "Total Records:"
        ws[f'A{row}'].font = Font(bold=True)
        ws[f'B{row}'] = len(df)

        # Column statistics
        for col in df.columns:
            row += 1
            if df[col].dtype in ['int64', 'float64']:
                # Numeric column statistics
                ws[f'A{row}'] = f"{col} (Average):"
                ws[f'A{row}'].font = Font(bold=True)
                ws[f'B{row}'] = round(df[col].mean(), 2) if not df[col].isna().all() else 0

                row += 1
                ws[f'A{row}'] = f"{col} (Total):"
                ws[f'A{row}'].font = Font(bold=True)
                ws[f'B{row}'] = round(df[col].sum(), 2) if not df[col].isna().all() else 0
            else:
                # Categorical column statistics
                unique_count = df[col].nunique()
                ws[f'A{row}'] = f"{col} (Unique Values):"
                ws[f'A{row}'].font = Font(bold=True)
                ws[f'B{row}'] = unique_count

        # Auto-adjust column widths
        ws.column_dimensions['A'].width = 25
        ws.column_dimensions['B'].width = 15

    def _add_charts(self, wb, ws, df: pd.DataFrame):
        """Add appropriate charts based on data characteristics."""

        ws.title = "Charts"
        chart_row = 2

        try:
            # Determine chart types based on data
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                # Create aggregated data for charting
                if len(df) > 20:  # For large datasets, aggregate data
                    chart_data = self._prepare_chart_data(df, numeric_cols, categorical_cols)
                else:
                    chart_data = df

                # Bar Chart - if we have categorical and numeric data
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    self._create_bar_chart(wb, ws, chart_data, categorical_cols[0], numeric_cols[0], chart_row)
                    chart_row += 18

                # Pie Chart - for categorical distribution
                if len(categorical_cols) > 0:
                    self._create_pie_chart(wb, ws, chart_data, categorical_cols[0], chart_row)
                    chart_row += 18

                # Line Chart - if we have time series data
                if self._has_date_column(df):
                    date_col = self._get_date_column(df)
                    if date_col and len(numeric_cols) > 0:
                        self._create_line_chart(wb, ws, chart_data, date_col, numeric_cols[0], chart_row)

        except Exception as e:
            logger.warning(f"Error creating charts: {e}")
            # Add a note about chart creation failure
            ws['A1'] = f"Chart creation encountered an issue: {str(e)}"
            ws['A1'].font = Font(color='FF0000')

    def _prepare_chart_data(self, df: pd.DataFrame, numeric_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
        """Prepare aggregated data for charting."""

        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            # Group by first categorical column and aggregate numeric columns
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]

            if cat_col in df.columns and num_col in df.columns:
                grouped = df.groupby(cat_col)[num_col].agg(['sum', 'count', 'mean']).reset_index()
                grouped.columns = [cat_col, f'{num_col}_sum', f'{num_col}_count', f'{num_col}_mean']
                return grouped

        return df.head(20)  # Return first 20 rows if aggregation fails

    def _create_bar_chart(self, wb, ws, df: pd.DataFrame, category_col: str, value_col: str, start_row: int):
        """Create a professional bar chart."""

        # Write chart data to worksheet
        ws.cell(start_row, 1, "Bar Chart Data")
        ws.cell(start_row, 1).font = Font(bold=True)

        # Write headers
        ws.cell(start_row + 2, 1, category_col)
        ws.cell(start_row + 2, 2, value_col)

        # Write data (limit to top 10 for readability)
        chart_data = df.head(10)
        for idx, (_, row) in enumerate(chart_data.iterrows()):
            ws.cell(start_row + 3 + idx, 1, str(row[category_col])[:20])  # Truncate long names
            ws.cell(start_row + 3 + idx, 2, float(row[value_col]) if pd.notna(row[value_col]) else 0)

        # Create chart
        chart = BarChart()
        chart.type = "col"
        chart.style = 10
        chart.title = f"{value_col} by {category_col}"
        chart.y_axis.title = value_col
        chart.x_axis.title = category_col

        # Set data range
        data_range = Reference(ws, min_col=2, min_row=start_row + 2, max_row=start_row + 2 + len(chart_data))
        categories = Reference(ws, min_col=1, min_row=start_row + 3, max_row=start_row + 2 + len(chart_data))

        chart.add_data(data_range, titles_from_data=True)
        chart.set_categories(categories)

        # Position chart
        ws.add_chart(chart, f"D{start_row}")

    def _create_pie_chart(self, wb, ws, df: pd.DataFrame, category_col: str, start_row: int):
        """Create a professional pie chart."""

        # Count occurrences of each category
        if category_col in df.columns:
            value_counts = df[category_col].value_counts().head(8)  # Top 8 categories
        else:
            return

        # Write chart data
        ws.cell(start_row, 1, "Pie Chart Data")
        ws.cell(start_row, 1).font = Font(bold=True)

        ws.cell(start_row + 2, 1, category_col)
        ws.cell(start_row + 2, 2, "Count")

        for idx, (category, count) in enumerate(value_counts.items()):
            ws.cell(start_row + 3 + idx, 1, str(category)[:20])
            ws.cell(start_row + 3 + idx, 2, int(count))

        # Create chart
        chart = PieChart()
        chart.title = f"Distribution of {category_col}"

        # Set data range
        data_range = Reference(ws, min_col=2, min_row=start_row + 2, max_row=start_row + 2 + len(value_counts))
        categories = Reference(ws, min_col=1, min_row=start_row + 3, max_row=start_row + 2 + len(value_counts))

        chart.add_data(data_range, titles_from_data=True)
        chart.set_categories(categories)

        # Position chart
        ws.add_chart(chart, f"D{start_row}")

    def _create_line_chart(self, wb, ws, df: pd.DataFrame, date_col: str, value_col: str, start_row: int):
        """Create a professional line chart for time series data."""

        # Sort by date and prepare data
        df_sorted = df.sort_values(date_col).head(20)

        # Write chart data
        ws.cell(start_row, 1, "Line Chart Data")
        ws.cell(start_row, 1).font = Font(bold=True)

        ws.cell(start_row + 2, 1, date_col)
        ws.cell(start_row + 2, 2, value_col)

        for idx, (_, row) in enumerate(df_sorted.iterrows()):
            ws.cell(start_row + 3 + idx, 1, str(row[date_col]))
            ws.cell(start_row + 3 + idx, 2, float(row[value_col]) if pd.notna(row[value_col]) else 0)

        # Create chart
        chart = LineChart()
        chart.title = f"{value_col} over {date_col}"
        chart.style = 12
        chart.y_axis.title = value_col
        chart.x_axis.title = date_col

        # Set data range
        data_range = Reference(ws, min_col=2, min_row=start_row + 2, max_row=start_row + 2 + len(df_sorted))
        categories = Reference(ws, min_col=1, min_row=start_row + 3, max_row=start_row + 2 + len(df_sorted))

        chart.add_data(data_range, titles_from_data=True)
        chart.set_categories(categories)

        # Position chart
        ws.add_chart(chart, f"D{start_row}")

    def _has_date_column(self, df: pd.DataFrame) -> bool:
        """Check if dataframe has date/datetime columns."""
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                return True
        return False

    def _get_date_column(self, df: pd.DataFrame) -> Optional[str]:
        """Get the first date column found."""
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                return col
        return None


# Global Excel exporter instance
excel_exporter = ExcelExporter()