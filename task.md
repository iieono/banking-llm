# Test Assignment (TZ) – AI Developer

## Project Overview
Create a Data Analyst Assistant prototype based on Local LLM for a banking system. Through this assignment, we will test your skills in AI, SQL, Data Engineering, and Visualization.

## Requirements

### 1. Database Creation
Create a mock database independently with **at least 1 million records**.

**Required Tables:**
- **Clients**
  - `id` - Primary key
  - `name` - Client name
  - `birth_date` - Date of birth
  - `region` - Geographic region

- **Accounts**
  - `id` - Primary key
  - `client_id` - Foreign key to Clients table
  - `balance` - Account balance
  - `open_date` - Account opening date

- **Transactions**
  - `id` - Primary key
  - `account_id` - Foreign key to Accounts table
  - `amount` - Transaction amount
  - `date` - Transaction date
  - `type` - Transaction type (debit/credit, transfer, etc.)

### 2. Local LLM Integration
- Use **Llama**, **Mistral**, or other **open-source models**
- When user writes a simple natural language prompt, the model should create SQL query
- The system must convert natural language to accurate SQL queries

**Example:**
- Input: "Show the total sum of transactions for Tashkent region in June 2024"
- Output: Generated SQL query that retrieves the requested data

### 3. Query Execution
- The generated SQL query must be executed on the database
- Handle query errors and edge cases
- Return results in a structured format
- Ensure query performance for large datasets

### 4. Excel Export with Visualizations
- Query results should be **automatically written to an Excel file**
- Create **diagrams/graphs** inside the Excel file:
  - Bar charts
  - Pie charts
  - Line charts (where appropriate)
- Format the Excel output professionally
- Include data tables alongside visualizations

### 5. User Interface
Write a **minimal CLI or Web interface** with the following flow:
- User enters natural language prompt
- System generates SQL query
- Query executes on database
- Results exported to Excel with charts
- User receives Excel file

### 6. Additional Features (Bonus)
- **Dockerfile**: Write Dockerfile to run the entire system inside a container
- **Web UI**: Create a simple frontend web interface
- **Advanced Features**: Additional functionality beyond basic requirements

## Expected Deliverables

### Core Deliverables
1. **Code**: Complete Python application with:
   - LLM integration
   - SQL query generation
   - Database operations
   - Excel export functionality

2. **README**: Comprehensive guide including:
   - Installation instructions
   - Usage examples
   - System requirements
   - Troubleshooting

3. **Demo**: Working demonstration showing:
   - Natural language prompt input
   - Excel output with graphics
   - End-to-end functionality

### Bonus Deliverables
4. **Dockerfile**: Container configuration for entire system
5. **Web Interface**: User-friendly web UI

## Evaluation Criteria

### Primary Assessment Points
✅ **SQL Query Generation Accuracy**: How well the system converts natural language to correct SQL
✅ **Code Structure and Cleanliness**: Proper architecture, readable code, best practices
✅ **Excel and Graphics Quality**: Professional output with meaningful visualizations
✅ **Local Operation Readiness**: System runs independently without external dependencies
✅ **Additional Features**: Docker implementation, UI quality, extra functionality

### Technical Requirements
- System must work with **local LLM** (no external API dependencies)
- Database must contain **1+ million records**
- Excel export must be **automated** and include **visualizations**
- Code must be **production-ready** and **well-documented**

## Timeline
**Deadline**: 4–5 days from assignment start

## Technical Specifications

### Suggested Technology Stack
- **Language**: Python
- **LLM**: Llama 3.x, Mistral 7B, or similar open-source model
- **Database**: SQLite (for simplicity) or PostgreSQL (for production-like scenario)
- **Excel Library**: openpyxl, xlsxwriter, or pandas
- **Visualization**: matplotlib, seaborn integrated with Excel
- **Web Framework**: Flask, FastAPI, or Streamlit (for bonus UI)
- **Containerization**: Docker

### Sample Queries to Handle
- "Show total transactions by region for the last month"
- "List top 10 clients by account balance"
- "Display monthly transaction trends for 2024"
- "Find accounts with balance above $50,000 in Tashkent"
- "Show transaction volume by type for each region"

### Performance Requirements
- Handle 1M+ record queries efficiently
- Generate SQL within 5 seconds
- Export to Excel within 10 seconds for typical queries
- Support concurrent users (for web interface)

### Security Considerations
- Validate generated SQL queries
- Prevent SQL injection
- Sanitize user inputs
- Secure database connections

## Success Criteria
A successful implementation will:
1. Generate accurate SQL from natural language prompts
2. Execute queries on large dataset efficiently
3. Produce professional Excel reports with meaningful charts
4. Provide intuitive user interface
5. Run entirely on local infrastructure
6. Include comprehensive documentation
7. Demonstrate clean, maintainable code architecture