import uuid, os
import re
import json
from dotenv import load_dotenv
import chainlit as cl
import pandas as pd
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from deep_research.graph import store_research_chain
from sql_agent.graph import sql_agent_chain

load_dotenv()

os.environ["LANGCHAIN_PROJECT"] = f"Banner Flip Engine - {uuid.uuid4().hex[0:8]}"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Define cache file path
CACHE_FILE = "cache/question_cache.json"

# Function to load cache from disk
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
    print(f"Cache file not found. Will create a new one at: {CACHE_FILE}")
    return {}

# Function to save cache to disk
def save_cache(cache):
    try:
        # Ensure directory exists
        cache_dir = os.path.dirname(CACHE_FILE)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        # Write cache to file
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"Error saving cache: {e}")

# Global cache variable
GLOBAL_CACHE = load_cache()
# Initialize with empty cache if loading failed
if GLOBAL_CACHE is None:
    GLOBAL_CACHE = {}

@tool("SQLAgent", description="Use this tool when you need to query information about competitor stores for Save-A-Lot. This tool can analyze store data to identify banner flip opportunities (converting competitor stores to Save-A-Lot stores). It's best used for questions about: store locations, distances between stores or to specific locations, store ownership information (independent vs. chain stores), parent company size, and other store attributes. The tool has special geospatial capabilities including distance calculation between coordinates and named locations. Example queries: 'Find independent stores within 10 miles of Chicago', 'How many competitor stores are in Florida?', 'Show stores owned by parent companies with fewer than 10 stores', 'List stores closest to Denver', 'Which independent stores in Tennessee might be good banner flip opportunities?' or 'Show all information for Smith's Grocery in Nashville'")
async def sql_agent_tool(input: str) -> str:
    result = await sql_agent_chain(input)
    return result

@tool("StoreResearch", description="Use this tool when you need comprehensive, detailed research about a specific competitor store for banner flip evaluation. This tool generates an in-depth report on a single store by first retrieving all available database information and then conducting external web research. The report includes market analysis, competitive landscape, ownership details, financial assessment, conversion feasibility, and recommendations. Structure your input as: 'provide all information for {specific store name/ID from user prompt}'. For example: 'provide all information for Smith's Grocery in Nashville' or 'provide all information for store ID #12345'. Only use for deep research on one specific store at a time, not for general queries about multiple stores or locations. Do not use for queries starting with 'Show all information for...' or 'Provide all information for...'")
async def store_research_tool(input: str) -> str:
    store_details = await sql_agent_chain(input)
    result = await store_research_chain(store_details)
    return result

system_message = """You are an assistant that helps users explore and analyze competitor store data.

EXTREMELY IMPORTANT: When tools return any output, you MUST preserve the EXACT text, formatting, and content in your response.
This includes:
1. NEVER summarize or change any information from tool responses
3. ALWAYS maintain ALL original text, markdown, and HTML exactly as provided by the tool

Present the information exactly as the tool provides it, without modification or omission.
"""

# Create a proper ChatPromptTemplate with system message
prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder(variable_name="messages")
])

# Create the agent with the proper prompt structure
app = create_react_agent(
    model=ChatOpenAI(model="gpt-4o", temperature=0),
    tools=[sql_agent_tool, store_research_tool],
    prompt=prompt,
    debug=True
)

def html_table_to_dataframe(html_table):
    """Convert an HTML table to a pandas DataFrame."""
    soup = BeautifulSoup(html_table, 'html.parser')
    table = soup.find('table')
    
    # Extract headers
    headers = []
    header_row = table.find('tr')
    for th in header_row.find_all('th'):
        headers.append(th.text.strip())
    
    # Extract rows
    rows = []
    for tr in table.find_all('tr')[1:]:  # Skip header row
        row = []
        for td in tr.find_all('td'):
            row.append(td.text.strip())
        rows.append(row)
    
    # Create DataFrame
    return pd.DataFrame(rows, columns=headers)

@cl.on_chat_start
async def start():
    cl.user_session.set("app", app)
    # Initialize session cache with the persistent global cache
    cl.user_session.set("cache", GLOBAL_CACHE)
    
    # Send welcome message
    welcome_message = """## Welcome to Save-A-Lot Banner Flip Engine

This tool helps you analyze competitor stores to identify the most promising banner flip opportunities.

### Sample Questions for Competitor Store Data:
- Find independent stores within 20 miles of Fort Lauderdale, FL
- Which stores in Cleveland, OH have parent companies with fewer than 5 locations?
- Show me all stores within 100 miles of Winchester distribution center sorted by distance

### Deep Store Research Example:
- Research Smith's Grocery in Nashville

You can query competitor store information or request in-depth research on specific stores.
"""
    await cl.Message(content=welcome_message).send()

# Function to process and render responses with tables if present
async def render_response(content):
    # Process HTML tables if present
    table_match = re.search(r'(<table>.*?</table>)\s*', content, re.DOTALL)
    
    if table_match:
        # Split into text content and table
        table = table_match.group(1)
        text_content = content[:table_match.start()].strip()
        
        try:
            # Convert table to DataFrame
            df = html_table_to_dataframe(table)
            
            # Send a single message with both text and table
            await cl.Message(
                content=text_content,
                elements=[cl.Dataframe(data=df, name="query_results", display="inline")]
            ).send()
        except Exception as e:
            # Fallback if conversion fails
            await cl.Message(content=content).send()
    else:
        # No table found, send as regular message
        await cl.Message(content=content).send()

@cl.on_message
async def handle(message: cl.Message):
    app = cl.user_session.get("app")
    cache = cl.user_session.get("cache")
    
    # Convert message content to lowercase for case-insensitive matching
    message_key = message.content.lower()
    
    # Check if the question is in cache (case insensitive)
    if message_key in cache:
        cached_response = cache[message_key]
        # Use the render_response function for cached responses
        await render_response(cached_response)
        return
    
    state = {"messages": [HumanMessage(content=message.content)]}
    response = await app.ainvoke(state)
    
    # Check if a tool was called in the response
    tool_output = None
    
    # Find the tool response message (typically the second-to-last message)
    for msg in reversed(response["messages"]):
        if hasattr(msg, 'name') and msg.name in ["SQLAgent", "StoreResearch"]:
            # Found a tool response, use it directly
            tool_output = msg.content
            break
    
    # If we have direct tool output, use it instead of the agent's summary
    if tool_output:
        content = tool_output
    else:
        # No tool was called, use the agent's regular response
        content = response["messages"][-1].content
    
    # Update both the session cache and global cache with lowercase key
    cache[message_key] = content
    GLOBAL_CACHE[message_key] = content
    # Persist the updated cache to disk
    save_cache(GLOBAL_CACHE)
    
    # Use the render_response function for new responses
    await render_response(content)

# async def handle(message: cl.Message):
#     app = cl.user_session.get("app")
#     state = {"messages" : [HumanMessage(content=message.content)]}
#     response = await app.ainvoke(state)
#     await cl.Message(content=response["messages"][-1].content).send()
