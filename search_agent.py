from google.adk.agents import Agent
from google.adk.tools import google_search

# Define the search agent
search_agent = Agent(
    name="search_assistant",
    model="gemini-pro",  # Or your preferred Gemini model (e.g., "gemini-2.0-flash")
    instruction="You are a helpful assistant. When the user asks a question that requires up-to-date information, use the 'google_search' tool to find relevant results and answer based on those results. If the question doesn't require real-time information, answer directly.",
    description="An agent that can search the web to answer questions.",
    tools=[google_search]
)

# Example interaction (requires a runtime)
# In a real application, you would use a runtime to send user queries to the agent
# and receive its responses. For now, we'll just print the agent's definition.

print(f"Agent Name: {search_agent.name}")
print(f"Agent Model: {search_agent.model}")
print(f"Agent Instruction: {search_agent.instruction}")
print(f"Agent Description: {search_agent.description}")
print(f"Agent Tools: {[tool.name for tool in search_agent.tools]}")