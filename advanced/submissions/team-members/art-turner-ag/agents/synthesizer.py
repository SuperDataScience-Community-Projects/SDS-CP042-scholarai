from .base_agent import BaseAgent

class SynthesizerAgent(BaseAgent):
    def run(self, topic, research_findings):
        # research_findings is a dict {subtopic: finding}
        
        findings_text = ""
        for subtopic, finding in research_findings.items():
            findings_text += f"## Sub-topic: {subtopic}\n{finding}\n\n"
            
        prompt = f"""
        Synthesize the following research findings into a comprehensive report on the main topic.
        
        Main Topic: {topic}
        
        Findings:
        {findings_text}
        
        The report should include:
        1. Executive Summary
        2. Key Insights by Sub-topic
        3. Conflicts or Gaps
        4. References (based on the provided sources)
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a synthesizer agent. Create a comprehensive report from research findings."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
