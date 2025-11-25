from .base_agent import BaseAgent
import json

class TopicSplitterAgent(BaseAgent):
    def run(self, topic):
        prompt = f"""
        Split the following research topic into 3 distinct, focused sub-topics for further research.
        Return the result as a JSON list of strings.
        
        Topic: {topic}
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant that splits topics into sub-topics."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        try:
            subtopics = json.loads(content)
            # Handle cases where the LLM might return a dict with a key like "subtopics"
            if isinstance(subtopics, dict):
                # Look for a list value
                for key, value in subtopics.items():
                    if isinstance(value, list):
                        return value
                # If no list found, return the keys? Or just fail gracefully.
                # Let's assume it might be {"subtopics": [...]}
                return list(subtopics.values())[0] if subtopics else []
            return subtopics
        except json.JSONDecodeError:
            return []
