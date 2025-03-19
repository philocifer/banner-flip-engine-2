from langgraph.graph import START,END, StateGraph
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
import sqlalchemy as sa
import pandas as pd
from typing import Annotated, Any, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_community.agent_toolkits.sql.prompt import SQL_PREFIX, SQL_SUFFIX
from langchain_core.tools import tool
from geopy.geocoders import Nominatim
from math import radians, sin, cos, sqrt, atan2
import operator
from sql_agent.geocoder import geocode_location
from operator import itemgetter

class SQLState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    input: str
    output: str

class SQLStateInput(TypedDict):
    input: str

class SQLStateOutput(TypedDict):
    output: str

def calculate_coordinate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two sets of geographic coordinates using the Haversine formula.
    
    Args:
        lat1: Latitude of the first point
        lon1: Longitude of the first point
        lat2: Latitude of the second point
        lon2: Longitude of the second point
        
    Returns:
        Distance in miles or None if coordinates are invalid
    """
    # Handle None values from database or geocoder
    if None in (lat1, lon1, lat2, lon2):
        return None
    
    try:
        # Haversine formula to calculate distance in miles
        R = 3958.8  # Earth radius in miles
        
        # Convert all coordinates to floats
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
        
        # Validate coordinate ranges
        if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90 and
                -180 <= lon1 <= 180 and -180 <= lon2 <= 180):
            return None

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    except (TypeError, ValueError) as e:
        print(f"Distance calculation error: {str(e)}")
        return None

def calculate_distance_to_location(lat1, lon1, location):
    """
    Calculate distance between coordinates and a text location.
    
    Args:
        lat1: Latitude of the first point
        lon1: Longitude of the first point
        location: Text location (address, city, etc.) to geocode
        
    Returns:
        Distance in miles or None if geocoding fails
    """
    # Geocode the location to get coordinates
    geocoded = geocode_location(location)
    if not geocoded:
        return None
        
    # Extract coordinates from the tuple and calculate distance
    lat2, lon2 = geocoded
    return calculate_coordinate_distance(lat1, lon1, lat2, lon2)

sql_llm = ChatOpenAI(
    temperature=0, 
    model="gpt-4o"
)

# Configure engine for thread-safe in-memory database
engine = sa.create_engine(
    "sqlite:///:memory:?cache=shared",
    poolclass=sa.pool.QueuePool,  # Use QueuePool for thread safety
    connect_args={'check_same_thread': False},  # Allow cross-thread connections
    echo=True
)

# Add this distance function to SQL
@sa.event.listens_for(engine, "connect")
def create_sqlite_functions(dbapi_connection, connection_record):
    dbapi_connection.create_function("distance", 4, calculate_coordinate_distance, deterministic=True)
    dbapi_connection.create_function("distance_to_location", 3, calculate_distance_to_location, deterministic=True)

csv_file = 'data/competitor_store_viable_with_dc.csv'
table_name = 'competitor_stores'

df = pd.read_csv(csv_file, encoding='utf-8')

with engine.begin() as conn:
    df.to_sql(
        table_name, 
        conn, 
        index=False, 
        if_exists='replace'
    )

db = SQLDatabase(engine)
toolkit = SQLDatabaseToolkit(
    db=db, 
    llm=sql_llm
)

SQL_PREFIX = """You are an agent designed to interact with a SQL database.  The database contains data about competitor stores for Save-A-Lot.  The goal is to identify competitor stores that are banner flip opportunities. A banner flip is when a competitor store is converted to a Save-A-Lot store.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Use "competitor stores" instead of "stores" in your answer.
If the query results is more than one row, your answer should always be in HTML table format
If the query results is a single row, do not use a table format.
If your answer includes an HTML table, the table must be the last element in your answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results and include that the results are limited to {top_k} in your answer.
If you order the results, include how the results are ordered in your answer.
You can order the results by a relevant column to return the most interesting examples in the database.
If the question asks for all information about a specific store, you should query all columns in the table and your answer must include all column values in your answer. Otherwise, only ask for the relevant columns given the question, but always include at least the following: store name, address, city, state.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

The database has two special distance functions:
1. distance(lat1, lon1, lat2, lon2) - Calculates miles between two coordinate pairs.
   Example: SELECT * FROM competitor_stores WHERE distance(latitude, longitude, 39.7392, -104.9903) < 10;

2. distance_to_location(lat1, lon1, 'location_name') - Calculates miles between a coordinate and a named location.
   Example: SELECT COUNT(*) FROM competitor_stores WHERE distance_to_location(latitude, longitude, 'Miami, FL') < 20;
   Example: SELECT store_name, street_address, city_name, state_iso2_code, postal_code FROM competitor_stores WHERE distance_to_location(latitude, longitude, 'Chicago') < 15 ORDER BY distance_to_location(latitude, longitude, 'Chicago');

Always prefer to use distance_to_location when user asks about stores near a city or named location, as it handles geocoding automatically.

IMPORTANT COLUMN METADATA:
- independent_store_indicator: Indicates if a store is independent or part of a chain. Valid values are "Independent Store" or "Chain Store".
- parent_company_store_count: The range of the number of stores in the parent company. Valuid values are "1 Store", "2-3 Stores", "4-5 Stores", "6-10 Stores", "11-25 Stores", "26-50 Stores", "51-100 Stores", "101-200 Stores", "201-500 Stores".
- closest_dc_name: The name of the distribution center closest to the store.
- closest_dc_distance_miles: The distance in miles between the store and the closest distribution center.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.
"""

sql_agent = create_sql_agent(
    llm=sql_llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="tool-calling",
    prefix=SQL_PREFIX,
    suffix=SQL_SUFFIX,
    handle_parsing_errors=True
)

async def query_competitor_stores(state: SQLState):
    result = await sql_agent.ainvoke({"input": state["input"]})
    output = result.get("output", "No results found")
    return {"output": output, "messages": [HumanMessage(content=output, name="QueryCompetitorStores")]}

builder = StateGraph(SQLState, input=SQLStateInput, output=SQLStateOutput)
builder.add_node("QueryCompetitorStores", query_competitor_stores)
builder.add_edge(START, "QueryCompetitorStores")
builder.add_edge("QueryCompetitorStores", END)

sql_graph = builder.compile()

async def sql_agent_chain(message: str):
    initial_state = {"input": message, "messages": [HumanMessage(content=message, name="User")]}
    result = await sql_graph.ainvoke(initial_state)
    return result["output"]
