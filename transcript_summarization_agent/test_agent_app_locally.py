from vertexai.preview import reasoning_engines

from agent import root_agent

agent_app = reasoning_engines.AdkApp(
    agent=root_agent,
    enable_tracing=True,
)

session = agent_app.create_session(user_id="u_123")

for event in agent_app.stream_query(
    user_id="u_123",
    session_id=session.id,
    message="""
            Virtual Agent: Hi, I am a vehicle sales agent. How can I help you?
            User: I'd like to buy a car.
            Virtual Agent: Can I interest you in a boat?
            User: No, a car. 
            Virtual Agent: This boat will be $10,000.
            User: Goodbye.
        """,
):
    print(event["content"]["parts"][0]["text"])