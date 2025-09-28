"""Main application entry point for BankingLLM system."""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Main entry point with mode selection."""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == "web" or mode == "gradio":
            # Launch Gradio web interface
            print("Starting Banking Analysis Web Interface...")
            from gradio_app import main as gradio_main
            gradio_main()

        elif mode == "cli":
            # Launch CLI interface
            print("Starting Banking Analysis CLI...")
            from cli import BankAICLI
            app = BankAICLI()
            app.interactive_mode()

        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python main.py [web|cli]")
            print("  web/gradio - Launch web interface (default)")
            print("  cli        - Launch command line interface")
            sys.exit(1)
    else:
        # Default to web interface
        print("Starting Banking Analysis Web Interface...")
        print("Use 'python main.py cli' for command line interface")
        from gradio_app import main as gradio_main
        gradio_main()

if __name__ == "__main__":
    main()