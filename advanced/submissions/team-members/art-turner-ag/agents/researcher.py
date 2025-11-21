from .base_agent import BaseAgent
from tavily import TavilyClient
import os

class ResearcherAgent(BaseAgent):
    def __init__(self, model="gpt-4o"):
        super().__init__(model)
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def run(self, subtopic):
        # 1. Search using Tavily
        search_result = self.tavily.search(query=subtopic, search_depth="advanced")
        results = search_result.get("results", [])
        
        context = "\n\n".join([f"Source: {r['url']}\nContent: {r['content']}" for r in results[:3]])
        
        # 2. Summarize findings
        prompt = f"""
        Research the following sub-topic using the provided search results.
        Provide a detailed summary including key insights and citations.
        
        Sub-topic: {subtopic}
        
        Search Results:
        {context}
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a researcher agent. Summarize findings based on search results."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
