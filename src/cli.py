"""Command Line Interface for Bank AI LLM system."""

import sys
from pathlib import Path
from typing import Optional

import click
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table

from .config import settings
from .database import db_manager
from .excel_export import excel_exporter
from .llm_service import llm_service

# Initialize rich console
console = Console()


class BankAICLI:
    """Main CLI application class."""

    def __init__(self):
        self.console = console
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for CLI output."""
        logger.remove()  # Remove default logger
        logger.add(
            sys.stderr,
            level=settings.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )

    def display_welcome(self):
        """Display welcome message and instructions."""
        welcome_text = """
🏦 Bank AI LLM Data Analyst Assistant

Transform natural language queries into SQL and get professional Excel reports!

Available commands:
• query - Run a natural language query
• samples - Show sample queries you can try
• stats - Display database statistics
• setup - Initialize database with mock data
• exit - Exit the application
        """

        panel = Panel.fit(
            welcome_text,
            title="Welcome",
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(panel)

    def display_stats(self):
        """Display database statistics."""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task = progress.add_task("Loading database statistics...", total=None)
                stats = db_manager.get_database_stats()
                progress.update(task, completed=True)

            # Create statistics table
            table = Table(title="Database Statistics", style="cyan")
            table.add_column("Metric", style="bold")
            table.add_column("Value", justify="right")

            table.add_row("Total Clients", f"{stats['clients']:,}")
            table.add_row("Total Accounts", f"{stats['accounts']:,}")
            table.add_row("Total Transactions", f"{stats['transactions']:,}")
            table.add_row("Regions", ", ".join(stats['regions']))

            if stats['date_range']['first_transaction']:
                table.add_row("Date Range", f"{stats['date_range']['first_transaction']} to {stats['date_range']['last_transaction']}")

            self.console.print(table)

        except Exception as e:
            self.console.print(f"[red]Error loading statistics: {e}[/red]")

    def display_sample_queries(self):
        """Display sample queries users can try."""
        samples = llm_service.suggest_sample_queries()

        table = Table(title="Sample Queries", style="green")
        table.add_column("#", width=3)
        table.add_column("Query", style="cyan")

        for i, query in enumerate(samples, 1):
            table.add_row(str(i), query)

        self.console.print(table)
        self.console.print("\n[dim]Tip: Copy and paste any query, or create your own![/dim]")

    def run_query(self, user_query: Optional[str] = None):
        """Run a natural language query and export results."""
        if not user_query:
            user_query = Prompt.ask("\n[bold cyan]Enter your query[/bold cyan]")

        if not user_query.strip():
            self.console.print("[yellow]Please enter a query.[/yellow]")
            return

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                # Generate SQL
                task1 = progress.add_task("Generating SQL query...", total=None)
                llm_result = llm_service.generate_sql(user_query)
                progress.update(task1, completed=True)

                if not llm_result['success']:
                    self.console.print(f"[red]Error: {llm_result.get('error', 'Failed to generate SQL')}[/red]")
                    return

                sql_query = llm_result['sql_query']

                # Display generated SQL
                self.console.print("\n[bold green]Generated SQL:[/bold green]")
                syntax = Syntax(sql_query, "sql", theme="monokai", line_numbers=True)
                self.console.print(Panel(syntax, title="SQL Query", border_style="green"))

                # Execute query
                task2 = progress.add_task("Executing query...", total=None)
                results = db_manager.execute_query(sql_query)
                progress.update(task2, completed=True)

                if not results:
                    self.console.print("[yellow]Query executed successfully but returned no results.[/yellow]")
                    return

                self.console.print(f"\n[green]Query returned {len(results)} rows[/green]")

                # Preview first few results
                if len(results) > 0:
                    self.display_results_preview(results)

                # Export to Excel
                task3 = progress.add_task("Exporting to Excel...", total=None)
                excel_path = excel_exporter.export_query_results(results, llm_result)
                progress.update(task3, completed=True)

                self.console.print(f"\n[bold green]✅ Excel report generated:[/bold green] {excel_path}")

                # Generate explanation
                explanation = llm_service.get_query_explanation(sql_query)
                if explanation:
                    self.console.print(f"\n[bold blue]Query Explanation:[/bold blue]\n{explanation}")

        except Exception as e:
            logger.error(f"Error running query: {e}")
            self.console.print(f"[red]Error: {e}[/red]")

    def display_results_preview(self, results, max_rows=5):
        """Display a preview of query results."""
        if not results:
            return

        # Create table for preview
        table = Table(title=f"Results Preview (showing {min(len(results), max_rows)} of {len(results)} rows)")

        # Add columns
        columns = list(results[0].keys())
        for col in columns:
            table.add_column(col, style="cyan")

        # Add rows
        for i, row in enumerate(results[:max_rows]):
            values = [str(row.get(col, '')) for col in columns]
            # Truncate long values
            values = [val[:30] + "..." if len(val) > 30 else val for val in values]
            table.add_row(*values)

        if len(results) > max_rows:
            table.add_row(*["..." for _ in columns], style="dim")

        self.console.print(table)

    def setup_database(self):
        """Initialize database with mock data."""
        self.console.print("[yellow]This will create a new database with 1M+ records. Continue? (y/N)[/yellow]")
        if not Prompt.ask("", default="n").lower().startswith('y'):
            return

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console
            ) as progress:
                task1 = progress.add_task("Creating database tables...", total=None)
                db_manager.create_tables()
                progress.update(task1, completed=True)

                task2 = progress.add_task("Generating mock data (this may take a few minutes)...", total=None)
                db_manager.generate_mock_data()
                progress.update(task2, completed=True)

            self.console.print("[bold green]✅ Database setup completed successfully![/bold green]")
            self.display_stats()

        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            self.console.print(f"[red]Error: {e}[/red]")

    def interactive_mode(self):
        """Run in interactive mode."""
        self.display_welcome()

        while True:
            try:
                command = Prompt.ask("\n[bold]Enter command[/bold]", default="query").lower().strip()

                if command in ['exit', 'quit', 'q']:
                    self.console.print("[green]Goodbye! 👋[/green]")
                    break
                elif command in ['query', 'q']:
                    self.run_query()
                elif command in ['samples', 'sample', 's']:
                    self.display_sample_queries()
                elif command in ['stats', 'statistics']:
                    self.display_stats()
                elif command in ['setup', 'init']:
                    self.setup_database()
                elif command == 'help':
                    self.display_welcome()
                else:
                    self.console.print(f"[red]Unknown command: {command}[/red]")
                    self.console.print("[yellow]Available commands: query, samples, stats, setup, help, exit[/yellow]")

            except KeyboardInterrupt:
                self.console.print("\n[green]Goodbye! 👋[/green]")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                self.console.print(f"[red]Error: {e}[/red]")


# CLI Commands using Click
@click.group()
def cli():
    """Bank AI LLM Data Analyst Assistant - Transform natural language to SQL and Excel reports."""
    pass


@cli.command()
@click.option('--query', '-q', help='Natural language query to execute')
def query(query):
    """Run a natural language query."""
    app = BankAICLI()
    if query:
        app.run_query(query)
    else:
        app.run_query()


@cli.command()
def interactive():
    """Run in interactive mode."""
    app = BankAICLI()
    app.interactive_mode()


@cli.command()
def setup():
    """Initialize database with mock data."""
    app = BankAICLI()
    app.setup_database()


@cli.command()
def stats():
    """Display database statistics."""
    app = BankAICLI()
    app.display_stats()


@cli.command()
def samples():
    """Show sample queries."""
    app = BankAICLI()
    app.display_sample_queries()


if __name__ == '__main__':
    cli()