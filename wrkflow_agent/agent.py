import os
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool # import
from google.adk.tools.crewai_tool import CrewaiTool

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from crewai_tools import FileWriterTool

from google.adk.tools import exit_loop

load_dotenv()

model_name = os.getenv("MODEL")

# Tools

def append_to_state(
    tool_context: ToolContext,
    field: str,
    response: str
) -> dict[str, str]:
    """Append new output to an existing state key.
    
    Args:
        field (str): a field name to append to
        response (str): a string to append to the field
    
    Returns:
        dict[str, str]: {"status": "success"}
    """
    existing_state = tool_context.state.get(field, [])
    tool_context.state[field] = existing_state + [response]
    return {"status": "success"}
    
    
# Agents

box_office_report = Agent(
    name="box_office_report",
    model=model_name,
    description="Considers the box office potential of this film",
    instruction="""
    TITLE:
    {{ title? }}

    STORY_SO_FAR:
    {{ story_so_far? }}

    INSTRUCTIONS:
    Write a report on the box office potential of a movie
    with a title of TITLE and a plot like that described in STORY_SO_FAR
    based on the reported box office performance of other recent films.
    """,
    output_key="box_office_report"
)

casting_agent = Agent(
    name="casting_agent",
    model=model_name,
    description="Generates casting ideas for this film",
    instruction="""
    TITLE:
    {{ title? }}

    STORY_SO_FAR:
    {{ story_so_far? }}

    INSTRUCTIONS:
    Generate ideas for casting for the characters described in STORY_SO_FAR
    based on who has done well with similar roles in the past.
    """,
    output_key="casting_report"
)

supplemental_reports = ParallelAgent(
    name="supplemental_reports",
    sub_agents=[box_office_report, casting_agent]
)

questioner = Agent(
    name="questioner",
    model=model_name,
    description="Asks good questions to prepare for a round of research",
    instruction="""
    PROMPT:
    {{ prompt? }}

    STORY_SO_FAR:
    {{ story_so_far? }}

    CRITIQUE:
    {{ critique? }}

    INSTRUCTIONS:
    Create a list of research questions based on:
    - Finding good historical characters related to the PROMPT
    - If there is a STORY_SO_FAR, what questions could help deepen
    its grounding on historical facts?
    - If there is a CRITIQUE, what research could address the
    improvements suggested there?

    Output only 5 questions.
    """,
    output_key="questions"
)

critic = Agent(
    name="critic",
    model=model_name,
    description="Offers a critique of the story so that it can be improved.",
    instruction="""
    INSTRUCTIONS:
    Offer critique of the following aspects of the STORY_SO_FAR:
    - Does it feel grounded in a real time period in history?
    - Does it sufficiently incorporate historical details from the RESEARCH?
    - Does it meet a satisfying three-act cinematic structure?
    - Do the characters' struggles seem engaging?
    If the STORY_SO_FAR does a good job with these questions,
    exit the writing loop with your 'exit_loop' tool.
    STORY_SO_FAR:
    {{ story_so_far? }}

    RESEARCH:
    {{ research? }}
    """,
    output_key="critique",
    tools=[exit_loop]
)

file_writer = Agent(
    name="file_writer",
    model=model_name,
    description="Save the movie outline to a file",
    instruction="""
    - Use your 'file_writer_tool' to create a new txt file named after the MOVIE_TITLE
    - Write to the 'movie_pitches' directory.
    - Set 'overwrite' to 'true'. 
    - The file should include:
        - The MOVIE_TITLE
        - Any description, synopsis, and plot outline from STORY_SO_FAR
    - Use your 'file_writer_tool' to create a new txt file named after the MOVIE_TITLE
    - Write to the 'movie_pitches' directory.
    - Set 'overwrite' to 'true'. 
    - The file should include:
        - The MOVIE_TITLE
        - Any description, synopsis, and plot outline from STORY_SO_FAR
        - The BOX_OFFICE_REPORT
        - The CASTING_REPORT

    MOVIE_TITLE: {{ title? }}

    STORY_SO_FAR:
    {{ story_so_far? }}

    BOX_OFFICE_REPORT:
    {{ box_office_report? }}

    CASTING_REPORT:
    {{ casting_report? }}
    
    MOVIE_TITLE: {{ title? }}

    STORY_SO_FAR:
    {{ story_so_far? }}
    """,
        # Add the CrewAI FileWriterTool below
    tools = [CrewaiTool(
        name="file_writer_tool",
        description=("Writes a file to disk when run with a filename,"
                     "content, overwrite, and an optional directory"),
        tool=FileWriterTool()
    )]
)

titler = Agent(
    name="titler",
    model=model_name,
    description="Writes a great movie title.",
    instruction="""
    INSTRUCTIONS:
    Provide a marketable, contemporary movie title suggestion
    for the movie described in the STORY_SO_FAR. If a title has been
    suggested in STORY_SO_FAR, you can use it, or replace it with
    a better one.
    Do not attempt to improve the story.
    Only provide 1 title.

    STORY_SO_FAR:
    {{ story_so_far? }}
    """,
    output_key="title",
    )

writer = Agent(
    name="writer",
    model=model_name,
    description="Write a plot outline about a historical character.",
    instruction="""
    INSTRUCTIONS:
    Your goal is to write a logline and three-act plot outline for an inspiring movie
    about the historical character(s) described by the prompt {{ prompt? }}, grounding
    it in history provided by the RESEARCH provided. If there is a STORY_SO_FAR,
    improve upon it. If there is CRITIQUE, use those thoughts to improve upon the
    outline.

    STORY_SO_FAR:
    {{ story_so_far? }}

    RESEARCH:
    {{ research? }}

    CRITIQUE:
    {{ critique? }}
    """,
    output_key="story_so_far",
)

researcher = Agent(
    name="researcher",
    model=model_name,
    description="Answer research questions using Wikipedia.",
    instruction="""
    INSTRUCTIONS:
    - Use your Wikipedia tool to search Wikipedia to conduct research
      based on the PROMPT and QUESTIONS.
    - Use the 'append_to_state' tool to add your research
      to the field 'research'.

    PROMPT:
    {{ prompt? }}

    QUESTIONS:
    {{ questions? }}
    """,
    # Add the LangChain Wikipedia tool below
    tools = [LangchainTool(
                tool=WikipediaQueryRun(
                        api_wrapper=WikipediaAPIWrapper()
                )
            ),
            append_to_state]
)

writers_room = LoopAgent(
    name="writers_room",
    description="Iterates through research and writing to improve a movie plot outline.",
    sub_agents = [questioner,
                researcher,
                writer,
                critic],
    max_iterations=5
    )

story_team = SequentialAgent(
    name="story_team",
    description="Writes a film plot, titles it, and saves it.",
    sub_agents=[
              writers_room,
              titler,
              supplemental_reports,
              file_writer
            ]
)
#researcher,
#writer,

root_agent = Agent(
    name="greeter",
    model=model_name,
    description="Guides the user in crafting a movie plot.",
    instruction="""
    - Let the user know you will help them write a pitch for a hit movie. Ask them for   
      a specific historical figure that the movie should be about.
    - When they respond:
      use the 'append_to_state' tool to set the value of 'prompt' to the user's input
      AND transfer to the 'story_team' agent
    - Let the user know you will help them write a pitch for a hit movie. Ask them for an occupation or other type of historical figure that the movie should be about.
    - When they respond, use the 'append_to_state' tool to set the value of 'prompt' to the user's input AND transfer to the 'story_team' agent
    """,
    tools=[append_to_state],
    sub_agents=[story_team]
)
