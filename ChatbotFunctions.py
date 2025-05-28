import json
from rapidfuzz import process, fuzz


class ChatbotFunctions:
    """
    A class containing all available functions that the LLM can call,
    along with their OpenAI-format descriptions for function calling.
    """

    # OpenAI format function descriptions
    __descriptions_all = {
        "get_matching_locations": {
            "name": "get_matching_locations",
            "description": "Get matching locations of FoodieSpot restaurants as per the location specified by the user. Use this function to take in the location specified by the user and display the most matching locations available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": "The area in Bengaluru the user wants to lookup the FoodieSpot joint in. e.g.: 'Koramangala', 'Whitefield' etc.",
                    }
                },
                "required": ["area"],
            },
        },
        "get_cuisine_by_area": {
            "name": "get_cuisine_by_area",
            "description": "Get the types of cuisine available at a particular area in Bengaluru. Use this function to develop recommendations for a user looking to dine at a spot in Bengaluru. Make sure the area you are looking for is exactly the same as the one you get from `get_matching_locations`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": "The area in Bengaluru the user wants to lookup the FoodieSpot joint in. e.g.: 'Koramangala', 'Whitefield' etc.",
                    }
                },
                "required": ["area"],
            },
        },
    }

    __restaurants_data = json.load(open("./data/restaurants_data.json", "r"))

    @classmethod
    def get_matching_locations(cls, area: str, top_n: int = 5) -> str:
        """
        Get random FoodieSpot restaurant locations around Bengaluru.

        Returns:
            str: Random restaurant locations
        """

        # Create list of tuples (area, place_dict)
        area_place_pairs = [
            (place["location"]["area"], place) for place in cls.__restaurants_data
        ]

        # Extract areas list for matching
        areas = [pair[0] for pair in area_place_pairs]

        # Get top N matches using RapidFuzz's extract with partial_ratio scorer
        top_matches = process.extract(
            area, areas, scorer=fuzz.partial_ratio, limit=top_n
        )

        results = []
        for matched_area, score, idx in top_matches:
            place = area_place_pairs[idx][1]
            results.append(place["location"]["area"])

        return f"Here is a list of the matching areas FoodieSpot is located in: {str(results)}\n\nList these options in a proper numbered fashion to the user and ask him to pick a number as a confirmation. Then, proceed with that area unless the user asks to search for another location."

    @classmethod
    def get_cuisine_by_area(cls, area: str) -> str:
        """
        Returns a list of unique cuisines found in the given area.

        Args:
            area (str): The name of the area/city to filter restaurants.

        Returns:
            str: A message containing unique cuisines found in the given location.
        """
        cuisines = set()

        for restaurant in cls.__restaurants_data:
            rest_location = restaurant.get("location", {}).get("area", "").lower()
            if area.lower() == rest_location:
                cuisine_field = restaurant.get("cuisine", [])
                cuisines.update(cuisine_field)

        return f"Here is a list of cuisines available at location {area}: {sorted(cuisines)}\n\nUse this list to show the various cuisine options available in an area. If the list is empty, ask the user to pick some other area."

    @classmethod
    def get_descriptions(cls):
        descriptions = cls.__descriptions_all.copy()

        return descriptions.values()
