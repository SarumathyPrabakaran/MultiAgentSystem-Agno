from flask import Flask, request, jsonify
from agno.agent import Agent
from agno.models.google import Gemini
from agno.models.groq import Groq
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import os

# Load environment variables (API keys)

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# -------------------------
# Define the Finance Agent Team
# -------------------------


web_agent=Agent(
    name="Web Agent",
    role="search the web for information",
    model=Groq(id="qwen-2.5-32b"),
    tools=[DuckDuckGoTools()],
    instructions="Always include the sources",
    show_tool_calls=True,
)


finance_agent = Agent(
    name="Finance Agent",
    role="Retrieve comprehensive financial data",
    model=Gemini(id="gemini-2.0-flash"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True,
                         stock_fundamentals=True, company_info=True)],
    instructions=(
        "You are a financial expert. Provide detailed, table-formatted financial data "
        "and analysis with appropriate sources."
        "Always include sources"
    ),
    show_tool_calls=True,
)
finance_team = Agent(
    team=[finance_agent],
    model=Groq(id="qwen-2.5-32b"),
    instructions=(
        "You are a team of finance experts. When given a query related to financial markets, "
        "provide comprehensive data and analysis using tables and include all relevant sources."
    ),
    show_tool_calls=True,
)

# -------------------------
# Define the Essay Writing Agent Team
# -------------------------
essay_agent = Agent(
    name="Essay Agent",
    role="Draft detailed essays based on input",
    model=Groq(id="qwen-2.5-32b"),
    instructions=(
        "You are an expert essay writer. Draft clear, comprehensive essays with headings, analysis, and conclusions. "
        "Include sources as needed."
    ),
    show_tool_calls=True,

)
essay_team = Agent(
    team=[essay_agent],
    model=Groq(id="qwen-2.5-32b"),
    instructions=(
        "You are a team of essay writers. When given a query or data, produce a well-structured and detailed essay. "
        "Ensure that all relevant sources are included in your final response."
        "Always include sources"
        "Prepare a final table of comparison for easy understanding"
    ),
    show_tool_calls=True,

)


agent_team=Agent(
    team=[web_agent,finance_agent, essay_agent],
    model=Groq(id="qwen-2.5-32b"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
)

# -------------------------
# Define the Autonomous Orchestrator Agent
# -------------------------


overall_agent = Agent(
    name="Autonomous Orchestrator",
    role="Coordinate and delegate tasks among expert teams autonomously",
    model=Gemini(id="gemini-2.0-flash"),
    instructions=(
        "You are an autonomous orchestrator with access to two independent expert teams: a Finance Team and an Essay Team. "
        "Upon receiving a query, analyze it and decide which expert team to consult. "
        "If the query requires financial data, first engage the Finance Team to retrieve the necessary information and then, "
        "if needed, pass that data to the Essay Team to draft a comprehensive essay. "
        "If the query is general or essay-based, delegate directly to the Essay Team. "
        "Communicate with the teams as needed and ensure the final answer includes sources. "
        "Do not use explicit conditional logic—make the decision autonomously."
        "Always include sources"
    ),
    show_tool_calls=True,
    team=[agent_team]
)

# -------------------------
# Set Up the Flask App and Endpoint
# -------------------------
app = Flask(__name__)


@app.route('/query', methods=['POST'])
def query_endpoint():
    data = request.get_json()
    query = data.get('query', '')
    
    # The overall agent handles the query autonomously.
    overall_response = overall_agent.run(query)
    return jsonify({"response": str(overall_response)})


if __name__ == '__main__':
    app.run(debug=True)
