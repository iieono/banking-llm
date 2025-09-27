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
from .filename_generator import filename_generator
from .database import format_currency, tiyin_to_sum


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

        # Multilingual labels
        self.labels = {
            'en': {
                'title': 'Banking Report',
                'data': 'Data',
                'charts': 'Charts',
                'summary': 'Summary',
                'generated': 'Generated on',
                'query': 'Query',
                'records': 'Records'
            },
            'ru': {
                'title': 'Банковский отчет',
                'data': 'Данные',
                'charts': 'Графики',
                'summary': 'Сводка',
                'generated': 'Создан',
                'query': 'Запрос',
                'records': 'Записей'
            },
            'uz': {
                'title': 'Banking hisoboti',
                'data': 'Ma\'lumotlar',
                'charts': 'Diagrammalar',
                'summary': 'Xulosa',
                'generated': 'Yaratilgan',
                'query': 'So\'rov',
                'records': 'Yozuvlar'
            }
        }

    def _detect_language(self, user_query: str) -> str:
        """Simple language detection for Excel labels."""
        query_lower = user_query.lower()

        # Check for Cyrillic characters (Russian)
        if any('\u0400' <= char <= '\u04ff' for char in query_lower):
            return 'ru'

        # Check for common Uzbek terms
        uzbek_terms = ['mijoz', 'hisob', 'operatsiya', 'toshkent', 'samarqand', 'buxoro']
        if any(term in query_lower for term in uzbek_terms):
            return 'uz'

        # Default to English
        return 'en'

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
            # Use fast filename generator to create descriptive filename
            sql_query = query_info.get('sql_query', '')
            filename = filename_generator.generate_filename(sql_query, data)

        filepath = self.output_dir / filename
        logger.info(f"Exporting data to Excel: {filepath}")

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)

        # Detect language from original query for proper formatting
        query_language = self._detect_query_language(query_info.get('user_query', ''))

        # Format currency columns with language-specific formatting
        df = self._format_currency_columns(df, query_language)

        # Create workbook and worksheets
        wb = Workbook()
        ws_data = wb.active
        ws_data.title = "Data"
        ws_summary = wb.create_sheet("Summary")

        # Export main data
        self._write_data_sheet(ws_data, df, query_info, query_language)

        # Create summary and charts
        self._create_summary_sheet(ws_summary, df, query_info, query_language)

        # Add basic professional charts
        chart_sheet = wb.create_sheet("Charts")
        self._add_charts(wb, chart_sheet, df, query_language)

        # Save workbook
        wb.save(filepath)
        logger.info(f"Excel export completed: {filepath}")

        return str(filepath)

    def _detect_query_language(self, user_query: str) -> str:
        """Detect the primary language of the user query."""
        if not user_query:
            return "english"

        query_lower = user_query.lower()

        # Cyrillic characters indicate Russian
        cyrillic_count = sum(1 for char in query_lower if '\u0400' <= char <= '\u04ff')

        # Common Russian banking terms
        russian_terms = [
            'клиент', 'счет', 'баланс', 'операция', 'филиал', 'банк',
            'покажи', 'найди', 'все', 'сумма', 'общий'
        ]

        # Common Uzbek banking terms
        uzbek_terms = [
            'mijoz', 'hisob', 'balans', 'operatsiya', 'filial', 'bank',
            'ko\'rsat', 'toping', 'barchasi', 'miqdor', 'jami', 'so\'m'
        ]

        # Count term matches
        russian_matches = sum(1 for term in russian_terms if term in query_lower)
        uzbek_matches = sum(1 for term in uzbek_terms if term in query_lower)

        # Determine language
        if cyrillic_count > 0 or russian_matches > uzbek_matches:
            return "russian"
        elif uzbek_matches > 0:
            return "uzbek"
        else:
            return "english"

    def _format_currency_columns(self, df: pd.DataFrame, language: str = "english") -> pd.DataFrame:
        """Format currency columns from tiyin to proper currency display format."""

        # Common currency column patterns
        currency_patterns = [
            'amount', 'balance', 'limit', 'fee', 'salary', 'deposit',
            'withdrawal', 'payment', 'transfer', 'income', 'minimum',
            'maximum', 'overdraft', 'daily', 'monthly', 'annual'
        ]

        # Default currency display for zero amounts
        zero_display = "0.00 UZS" if language == "english" else (
            "0.00 сум" if language == "russian" else "0.00 so'm"
        )

        df_copy = df.copy()

        for col in df.columns:
            col_lower = str(col).lower()
            # Check if column contains currency amounts (likely in tiyin)
            if any(pattern in col_lower for pattern in currency_patterns):
                # Check if column contains numeric values that could be tiyin amounts
                if df[col].dtype in ['int64', 'float64'] and not df[col].isna().all():
                    # Convert tiyin to proper currency format for display
                    try:
                        df_copy[col] = df[col].apply(
                            lambda x: format_currency(int(x), language) if pd.notna(x) and x != 0 else zero_display
                        )
                    except (ValueError, TypeError):
                        # If conversion fails, keep original values
                        pass

        return df_copy

    def _write_data_sheet(self, ws, df: pd.DataFrame, query_info: Dict, language: str = "english"):
        """Write data to the main data sheet with professional formatting."""

        # Language-specific titles
        titles = {
            "english": {
                "title": "Bank Data Analysis Report",
                "query": "Query:",
                "sql": "Generated SQL:",
                "export_time": "Export Time:"
            },
            "russian": {
                "title": "Отчёт по банковским данным",
                "query": "Запрос:",
                "sql": "Сгенерированный SQL:",
                "export_time": "Время экспорта:"
            },
            "uzbek": {
                "title": "Bank ma'lumotlari tahlil hisoboti",
                "query": "So'rov:",
                "sql": "Yaratilgan SQL:",
                "export_time": "Eksport vaqti:"
            }
        }

        lang_titles = titles.get(language, titles["english"])

        # Add header with query information
        ws.merge_cells('A1:E1')
        ws['A1'] = lang_titles["title"]
        ws['A1'].font = Font(size=16, bold=True, color=self.colors['primary'])
        ws['A1'].alignment = Alignment(horizontal='center')

        # Add query info
        ws['A3'] = lang_titles["query"]
        ws['A3'].font = Font(bold=True, color=self.colors['text'])
        ws['B3'] = query_info.get('user_query', 'N/A')

        ws['A4'] = lang_titles["sql"]
        ws['A4'].font = Font(bold=True, color=self.colors['text'])
        ws['B4'] = query_info.get('sql_query', 'N/A')

        ws['A5'] = lang_titles["export_time"]
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
            column_letter = None
            for cell in column:
                try:
                    # Skip merged cells
                    if hasattr(cell, 'column_letter'):
                        column_letter = cell.column_letter
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                except:
                    pass
            if column_letter:
                adjusted_width = min(max_length + 2, 50)
                ws.column_dimensions[column_letter].width = adjusted_width

    def _create_summary_sheet(self, ws, df: pd.DataFrame, query_info: Dict, language: str = "english"):
        """Create a summary sheet with key statistics."""

        # Language-specific summary labels
        summary_labels = {
            "english": {
                "title": "Data Summary",
                "total_records": "Total Records:",
                "average": "Average",
                "total": "Total",
                "unique_values": "Unique Values"
            },
            "russian": {
                "title": "Сводка данных",
                "total_records": "Общее количество записей:",
                "average": "Среднее",
                "total": "Всего",
                "unique_values": "Уникальные значения"
            },
            "uzbek": {
                "title": "Ma'lumotlar xulosasi",
                "total_records": "Jami yozuvlar:",
                "average": "O'rtacha",
                "total": "Jami",
                "unique_values": "Noyob qiymatlar"
            }
        }

        labels = summary_labels.get(language, summary_labels["english"])

        # Title
        ws.merge_cells('A1:D1')
        ws['A1'] = labels["title"]
        ws['A1'].font = Font(size=14, bold=True, color=self.colors['primary'])
        ws['A1'].alignment = Alignment(horizontal='center')

        # Basic statistics
        row = 3
        ws[f'A{row}'] = labels["total_records"]
        ws[f'A{row}'].font = Font(bold=True)
        ws[f'B{row}'] = len(df)

        # Column statistics
        for col in df.columns:
            row += 1
            if df[col].dtype in ['int64', 'float64']:
                # Numeric column statistics (for non-currency columns)
                ws[f'A{row}'] = f"{col} ({labels['average']}):"
                ws[f'A{row}'].font = Font(bold=True)
                ws[f'B{row}'] = round(df[col].mean(), 2) if not df[col].isna().all() else 0

                row += 1
                ws[f'A{row}'] = f"{col} ({labels['total']}):"
                ws[f'A{row}'].font = Font(bold=True)
                ws[f'B{row}'] = round(df[col].sum(), 2) if not df[col].isna().all() else 0
            else:
                # Categorical column statistics (includes formatted currency columns)
                unique_count = df[col].nunique()
                ws[f'A{row}'] = f"{col} ({labels['unique_values']}):"
                ws[f'A{row}'].font = Font(bold=True)
                ws[f'B{row}'] = unique_count

        # Auto-adjust column widths
        ws.column_dimensions['A'].width = 25
        ws.column_dimensions['B'].width = 15

    def _add_charts(self, wb, ws, df: pd.DataFrame, language: str = "english"):
        """Add appropriate charts based on data characteristics."""

        # Language-specific sheet titles
        chart_titles = {
            "english": "Charts",
            "russian": "Графики",
            "uzbek": "Diagrammalar"
        }

        ws.title = chart_titles.get(language, "Charts")
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
                    self._create_bar_chart(wb, ws, chart_data, categorical_cols[0], numeric_cols[0], chart_row, language)
                    chart_row += 18

                # Pie Chart - for categorical distribution
                if len(categorical_cols) > 0:
                    self._create_pie_chart(wb, ws, chart_data, categorical_cols[0], chart_row, language)
                    chart_row += 18

                # Line Chart - if we have time series data
                if self._has_date_column(df):
                    date_col = self._get_date_column(df)
                    if date_col and len(numeric_cols) > 0:
                        self._create_line_chart(wb, ws, chart_data, date_col, numeric_cols[0], chart_row, language)

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

    def _create_bar_chart(self, wb, ws, df: pd.DataFrame, category_col: str, value_col: str, start_row: int, language: str = "english"):
        """Create a professional bar chart."""

        # Language-specific labels
        chart_labels = {
            "english": {"chart_data": "Bar Chart Data"},
            "russian": {"chart_data": "Данные столбчатой диаграммы"},
            "uzbek": {"chart_data": "Ustunli diagramma ma'lumotlari"}
        }

        labels = chart_labels.get(language, chart_labels["english"])

        # Write chart data to worksheet
        ws.cell(start_row, 1, labels["chart_data"])
        ws.cell(start_row, 1).font = Font(bold=True)

        # Write headers
        ws.cell(start_row + 2, 1, category_col)
        ws.cell(start_row + 2, 2, value_col)

        # Write data (limit to top 10 for readability)
        chart_data = df.head(10)
        for idx, (_, row) in enumerate(chart_data.iterrows()):
            ws.cell(start_row + 3 + idx, 1, str(row[category_col])[:20])  # Truncate long names
            ws.cell(start_row + 3 + idx, 2, float(row[value_col]) if pd.notna(row[value_col]) else 0)

        # Use previously defined labels and add chart-specific terms
        labels["by"] = {"english": "by", "russian": "по", "uzbek": "bo'yicha"}[language]

        chart = BarChart()
        chart.type = "col"
        chart.style = 10
        chart.title = f"{value_col} {labels['by']} {category_col}"
        chart.y_axis.title = value_col
        chart.x_axis.title = category_col

        # Set data range
        data_range = Reference(ws, min_col=2, min_row=start_row + 2, max_row=start_row + 2 + len(chart_data))
        categories = Reference(ws, min_col=1, min_row=start_row + 3, max_row=start_row + 2 + len(chart_data))

        chart.add_data(data_range, titles_from_data=True)
        chart.set_categories(categories)

        # Position chart
        ws.add_chart(chart, f"D{start_row}")

    def _create_pie_chart(self, wb, ws, df: pd.DataFrame, category_col: str, start_row: int, language: str = "english"):
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

    def _create_line_chart(self, wb, ws, df: pd.DataFrame, date_col: str, value_col: str, start_row: int, language: str = "english"):
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

    def export_to_excel(self, results_df: pd.DataFrame, filename: str, query_description: str, sql_query: str) -> str:
        """Export DataFrame to Excel with multilingual support."""
        # Detect language from query description
        language = self._detect_language(query_description)
        labels = self.labels[language]

        # Create workbook
        wb = Workbook()

        # Data sheet
        ws_data = wb.active
        ws_data.title = labels['data']
        self._write_data_sheet(ws_data, results_df, {
            'query': query_description,
            'sql': sql_query,
            'timestamp': datetime.now()
        }, language)

        # Charts sheet if data is suitable
        if len(results_df) > 1:
            ws_charts = wb.create_sheet(title=labels['charts'])
            self._add_charts(wb, ws_charts, results_df, language)

        # Summary sheet
        ws_summary = wb.create_sheet(title=labels['summary'])
        self._create_summary_sheet(ws_summary, results_df, {
            'query': query_description,
            'sql': sql_query,
            'timestamp': datetime.now()
        }, language)

        # Save file
        filepath = self.output_dir / filename
        wb.save(filepath)

        return str(filepath)


# Global Excel exporter instance
excel_exporter = ExcelExporter()