"""Command Line Interface for BankingLLM system."""

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

# Handle both module and direct execution
try:
    from .config import settings
    from .database import db_manager
    from .excel_export import excel_exporter
    from .llm_service import llm_service
except ImportError:
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from src.config import settings
    from src.database import db_manager
    from src.excel_export import excel_exporter
    from src.llm_service import llm_service

# Initialize rich console with Windows compatibility
console = Console(force_terminal=True, legacy_windows=False)


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
BankingLLM Data Analyst

Transform natural language queries into SQL and get professional Excel reports!

Available commands:
* query - Run a natural language query
* samples - Show sample queries you can try
* stats - Display database statistics
* setup - Generate GitHub-optimized regional databases (multiple files)
* exit - Exit the application
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
            self.console.print("[blue]Loading database statistics...[/blue]")
            stats = db_manager.get_database_stats()

            # Create statistics table
            table = Table(title="Database Statistics", style="cyan")
            table.add_column("Metric", style="bold")
            table.add_column("Value", justify="right")

            table.add_row("Total Clients", f"{stats['clients']:,}")
            table.add_row("Total Accounts", f"{stats['accounts']:,}")
            table.add_row("Total Transactions", f"{stats['transactions']:,}")
            table.add_row("Total Branches", f"{stats['branches']:,}")
            table.add_row("Total Products", f"{stats['products']:,}")
            table.add_row("Regions", ", ".join(stats['regions']))
            table.add_row("Account Types", ", ".join(stats['account_types']))
            table.add_row("Transaction Types", ", ".join(stats['transaction_types']))

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
            # Generate SQL
            self.console.print("[blue]Generating SQL query...[/blue]")
            llm_result = llm_service.generate_sql(user_query)

            if not llm_result['success']:
                self.console.print(f"[red]Error: {llm_result.get('error', 'Failed to generate SQL')}[/red]")
                return

            sql_query = llm_result['sql_query']

            # Display generated SQL
            self.console.print("\n[bold green]Generated SQL:[/bold green]")
            syntax = Syntax(sql_query, "sql", theme="monokai", line_numbers=True)
            self.console.print(Panel(syntax, title="SQL Query", border_style="green"))

            # Execute query
            self.console.print("[blue]Executing query...[/blue]")
            results = db_manager.execute_query(sql_query)

            if not results:
                self.console.print("[yellow]Query executed successfully but returned no results.[/yellow]")
                return

            self.console.print(f"\n[green]Query returned {len(results)} rows[/green]")

            # Preview first few results
            if len(results) > 0:
                self.display_results_preview(results)

            # Export to Excel
            self.console.print("[blue]Exporting to Excel...[/blue]")
            excel_path = excel_exporter.export_query_results(results, llm_result)

            self.console.print(f"\n[bold green]Excel report generated:[/bold green] {excel_path}")

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

    def setup_database(self, force=False, force_regenerate=False):
        """Initialize enhanced single database with 1.2M+ realistic transactions."""
        if not force:
            self.console.print(f"[yellow]Generate enhanced banking database with {settings.num_transactions:,} transactions? (y/N)[/yellow]")
            if not Prompt.ask("", default="n").lower().startswith('y'):
                self.console.print("Aborted!")
                return

        try:
            self.console.print("\n[bold cyan]Enhanced Banking Database Generation[/bold cyan]")

            # Display generation configuration
            table = Table(title="Database Configuration", style="cyan")
            table.add_column("Parameter", style="bold")
            table.add_column("Value", justify="right")

            table.add_row("Clients", f"{settings.num_clients:,}")
            table.add_row("Transactions", f"{settings.num_transactions:,}")
            table.add_row("Merchant Categories", f"{len(settings.merchant_categories)}")
            table.add_row("Velocity Patterns", f"{len(settings.transaction_velocity_patterns)}")
            table.add_row("Seasonal Cycles", f"{len(settings.seasonal_banking_cycles)}")

            self.console.print(table)

            # Generate database
            self.console.print("[blue]Generating enhanced database...[/blue]")
            db_manager.generate_mock_data(force_regenerate=True)

            self.console.print("\n[bold green]Enhanced database generated successfully![/bold green]")

            # Show final statistics
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
                    self.console.print("[green]Goodbye![/green]")
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
                self.console.print("\n[green]Goodbye![/green]")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                self.console.print(f"[red]Error: {e}[/red]")


# CLI Commands using Click
@click.group()
def cli():
    """BankingLLM Data Analyst - Transform natural language to SQL and Excel reports."""
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
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompt')
@click.option('--force-regenerate', is_flag=True, help='Force regeneration even if data exists')
def setup(yes, force_regenerate):
    """Generate GitHub-optimized regional databases with maximum realism."""
    app = BankAICLI()
    app.setup_database(force=yes, force_regenerate=force_regenerate)




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