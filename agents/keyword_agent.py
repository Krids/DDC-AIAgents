from agents.base_agent import BaseAgent

class KeywordAgent(BaseAgent):
    """
    Agent responsible for finding relevant keywords for a given topic using Google Trends (pytrends).
    """

    def run(self, topic: str, language: str = "en-US", *args, **kwargs):
        if not topic:
            raise ValueError("Topic must be provided.")

        try:
            from pytrends.request import TrendReq
            pytrends = TrendReq(hl=language, tz=360)
            pytrends.build_payload([topic], cat=0, timeframe='now 7-d', geo='', gprop='')

            related_queries = pytrends.related_queries()
            results = related_queries.get(topic, {}).get('top', None)

            if results is not None and 'query' in results:
                keywords = results['query'].tolist()
                if keywords:
                    return keywords

        except Exception as e:
            # Optional: log the exception here for debugging
            print(f"[KeywordAgent] Warning: {e}")

        # Fallback: always return at least the topic itself
        return [topic]
