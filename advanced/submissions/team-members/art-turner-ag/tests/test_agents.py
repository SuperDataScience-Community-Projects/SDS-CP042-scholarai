import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path to import agents
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.topic_splitter import TopicSplitterAgent
from agents.researcher import ResearcherAgent
from agents.synthesizer import SynthesizerAgent

class TestAgents(unittest.TestCase):
    @patch('agents.base_agent.OpenAI')
    def test_topic_splitter(self, mock_openai):
        # Mock OpenAI response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '["Subtopic 1", "Subtopic 2", "Subtopic 3"]'
        mock_client.chat.completions.create.return_value = mock_response
        
        agent = TopicSplitterAgent()
        result = agent.run("Test Topic")
        
        self.assertEqual(result, ["Subtopic 1", "Subtopic 2", "Subtopic 3"])

    @patch('agents.base_agent.OpenAI')
    @patch('agents.researcher.TavilyClient')
    def test_researcher(self, mock_tavily, mock_openai):
        # Mock Tavily response
        mock_tavily_client = MagicMock()
        mock_tavily.return_value = mock_tavily_client
        mock_tavily_client.search.return_value = {
            "results": [
                {"url": "http://example.com", "content": "Example content"}
            ]
        }
        
        # Mock OpenAI response
        mock_openai_client = MagicMock()
        mock_openai.return_value = mock_openai_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Summary of findings"
        mock_openai_client.chat.completions.create.return_value = mock_response
        
        agent = ResearcherAgent()
        result = agent.run("Subtopic 1")
        
        self.assertEqual(result, "Summary of findings")

    @patch('agents.base_agent.OpenAI')
    def test_synthesizer(self, mock_openai):
        # Mock OpenAI response
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Final Report"
        mock_client.chat.completions.create.return_value = mock_response
        
        agent = SynthesizerAgent()
        result = agent.run("Test Topic", {"Subtopic 1": "Finding 1"})
        
        self.assertEqual(result, "Final Report")

if __name__ == '__main__':
    unittest.main()
