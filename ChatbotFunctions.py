class ChatbotFunctions:
    """
    A class containing all available functions that the LLM can call,
    along with their OpenAI-format descriptions for function calling.
    """

    # OpenAI format function descriptions
    __descriptions_all = {
        "get_weather": {
            "name": "get_weather",
            "description": "Get the current weather information for a specific city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city the user wants to get weather for, e.g. 'New York', 'London', 'Tokyo'",
                    }
                },
                "required": ["city"],
            },
        }
    }

    @staticmethod
    def get_weather(city: str) -> str:
        """
        Get weather information for a specific city

        Args:
            city (str): Name of the city

        Returns:
            str: Weather information
        """

        return f"""{city} Weather Report:
        Current temperature: 37°C
        Maximum: 40°C
        Minimum: 28°C
        AQI: 60 (satisfactory)
        Moisture: 48%
        UV index: Very high"""

    @classmethod
    def get_descriptions(cls):
        descriptions = cls.__descriptions_all.copy()

        return descriptions.values()
