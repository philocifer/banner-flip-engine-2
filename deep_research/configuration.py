from itertools import tee
import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict 

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from dataclasses import dataclass

DATA_SOURCES = """
- US Census Bureau (American Community Survey) for demographic information
- County property records for ownership, property size, and assessed values
- Business license databases through city/county websites
- SEC filings if the competitor is publicly traded or owned by a public company
- Local economic development reports from municipal websites
- Google Maps/Earth for store location, size estimation, parking assessment
- Street View for exterior condition assessment
- Business listings (Google Business Profile, Yelp) for operating hours, reviews
- Local news archives for articles about store closings, financial troubles
- Social media for customer sentiment and store activity
- LinkedIn for owner/decision maker information
- UCC filings (Uniform Commercial Code) which can indicate loans against business assets
- Tax lien records which may show financial distress
- Building permits showing recent renovations or lack thereof
- Health department inspections (often public record)
- Corporate registration information through Secretary of State websites
"""

DEFAULT_REPORT_STRUCTURE = f"""Your goal is to create a report that helps the Save-A-Lot business development team decide whether competitor grocery stores are banner flip opportunities. A banner flip is when a competitor store converts to a Save-A-Lot store: operating under the Save-A-Lot banner while maintaining current ownership, sourcing products from Save-A-Lot distribution centers, and conforming to Save-A-Lot store design standards. Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   
3. Conclusion
   - Aim for 1 structural element (either a list of table) that distills the main body sections 
   - Provide a concise summary of the report focusing on whether competitor grocery stores are banner flip opportunities

Aspects to include:
- Trade area demographics (income levels, population density, ethnic diversity)
- Competitive landscape
- Store visibility and accessibility
- Neighborhood economic health
- Building size and condition (exterior assessment)
- General layout compatibility with Save A Lot's format
- Current ownership structure
    - MANDATORY: Name of the owner/decision maker and their contact information (address, phone number, email)
    - Owner's willingness to convert
    - If the store is part of a chain, owner's willingness to convert all stores in the chain
- Business longevity
- Signs of financial distress
- Recent changes in operations
- Supplier relationships

Data sources to use:
{DATA_SOURCES}

Save-A-Lot information:
- Discount grocery model
- Offers a limited assortment of products
- Focuses on affordability with private label products
- Emphasizes value and convenience
- Average annual sales per store: $12.8 million
- Average store size: 16,000 square feet
"""

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"
    EXA = "exa"
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    LINKUP = "linkup"
    DUCKDUCKGO = "duckduckgo"

class PlannerProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"

class WriterProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    report_structure: str = DEFAULT_REPORT_STRUCTURE # Defaults to the default report structure
    number_of_queries: int = 5 # Number of search queries to generate per iteration
    max_search_depth: int = 3 # Maximum number of reflection + search iterations
    planner_provider: PlannerProvider = PlannerProvider.OPENAI  # Defaults to Anthropic as provider
    planner_model: str = "gpt-4o" # Defaults to claude-3-7-sonnet-latest
    planner_chat_model: BaseChatModel = None # Set this if you want to specify the Planner model directly. planner_provider and planner_model will be ignored.
    writer_provider: WriterProvider = WriterProvider.OPENAI # Defaults to Anthropic as provider
    writer_model: str = "gpt-4o" # Defaults to claude-3-5-sonnet-latest
    writer_chat_model: BaseChatModel = None # Set this if you want to specify the Writer model directly. writer_provider and writer_model will be ignored.
    search_api: SearchAPI = SearchAPI.TAVILY # Default to TAVILY
    search_api_config: Optional[Dict[str, Any]] = None 
    data_sources: str = DATA_SOURCES

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})
