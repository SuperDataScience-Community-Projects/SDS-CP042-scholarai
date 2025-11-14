"""Web search tool using Tavily API."""

import os
from typing import List, Dict, Optional
from tavily import TavilyClient


class WebSearchTool:
    """Wrapper for Tavily web search API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the web search tool.

        Args:
            api_key: Tavily API key. If not provided, will use TAVILY_API_KEY from environment.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")

        self.client = TavilyClient(api_key=self.api_key)

    def search(
        self,
        query: str,
        max_results: int = 10,
        search_depth: str = "advanced",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Search the web for relevant sources.

        Args:
            query: The search query
            max_results: Maximum number of results to return (default: 10)
            search_depth: "basic" or "advanced" search (default: "advanced")
            include_domains: Optional list of domains to include
            exclude_domains: Optional list of domains to exclude

        Returns:
            List of dictionaries with keys: 'title', 'url', 'snippet', 'score'
        """
        try:
            # Perform search with Tavily
            response = self.client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )

            # Extract and format results
            results = []
            for result in response.get("results", []):
                results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "snippet": result.get("content", ""),
                    "score": result.get("score", 0.0),
                })

            return results

        except Exception as e:
            raise RuntimeError(f"Web search failed: {str(e)}")


def web_search(query: str, k: int = 10) -> List[Dict[str, str]]:
    """
    Convenience function for web search.

    Args:
        query: The search query
        k: Number of results to return (default: 10)

    Returns:
        List of dictionaries with keys: 'title', 'url', 'snippet', 'score'
    """
    tool = WebSearchTool()
    return tool.search(query, max_results=k)
