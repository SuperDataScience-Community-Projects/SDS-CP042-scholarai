"""Research Agent using OpenAI Agents SDK."""

import os
import json
from typing import List, Dict, Optional
from openai import OpenAI
from tools.web_search import web_search


class ResearchAgent:
    """Agent that searches the web and curates relevant sources."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        max_sources: int = 10,
    ):
        """
        Initialize the Research Agent.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY from environment.
            model: OpenAI model to use
            max_sources: Maximum number of sources to fetch
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_sources = max_sources

        # Define the web search function for the agent
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for relevant academic sources, articles, and papers on a given topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for finding relevant sources",
                            },
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 10)",
                                "default": 10,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

        # System prompt for the research agent
        self.system_prompt = """You are a research assistant tasked with finding and curating relevant sources on a given topic.

Your responsibilities:
1. Use the web_search function to find relevant sources
2. Analyze the search results for relevance and quality
3. Curate the most relevant and reliable sources
4. Provide a structured summary of the sources found

When searching:
- Formulate clear, specific search queries
- Look for academic sources, reputable publications, and authoritative content
- Prioritize recent and well-cited sources
- Consider multiple perspectives on the topic

Return your findings in a structured format with the curated sources."""

    def research(self, topic: str) -> Dict:
        """
        Research a topic by searching the web and curating sources.

        Args:
            topic: The research topic or question

        Returns:
            Dictionary containing curated sources and metadata
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Research this topic and find the most relevant sources: {topic}",
            },
        ]

        # Initial call to the model
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
        )

        # Process tool calls
        sources = []
        while response.choices[0].message.tool_calls:
            # Add assistant's response to messages
            messages.append(response.choices[0].message)

            # Execute each tool call
            for tool_call in response.choices[0].message.tool_calls:
                if tool_call.function.name == "web_search":
                    # Parse arguments
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query")
                    k = args.get("k", self.max_sources)

                    # Execute search
                    search_results = web_search(query, k=k)
                    sources.extend(search_results)

                    # Add tool response to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(search_results),
                    })

            # Get next response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )

        # Get final analysis from the agent
        final_message = response.choices[0].message.content

        return {
            "topic": topic,
            "sources": sources,
            "analysis": final_message,
            "total_sources": len(sources),
        }

    def curate_sources(
        self, sources: List[Dict], top_n: int = 5
    ) -> List[Dict]:
        """
        Curate and rank sources by relevance.

        Args:
            sources: List of source dictionaries
            top_n: Number of top sources to return

        Returns:
            List of top N curated sources
        """
        # Sort by score (if available) and return top N
        sorted_sources = sorted(
            sources,
            key=lambda x: x.get("score", 0.0),
            reverse=True
        )
        return sorted_sources[:top_n]


def create_research_agent(
    model: str = "gpt-4-turbo-preview",
    max_sources: int = 10
) -> ResearchAgent:
    """
    Factory function to create a research agent.

    Args:
        model: OpenAI model to use
        max_sources: Maximum number of sources to fetch

    Returns:
        Initialized ResearchAgent instance
    """
    return ResearchAgent(model=model, max_sources=max_sources)
