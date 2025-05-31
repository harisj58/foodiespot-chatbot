import json
from rapidfuzz import process, fuzz
import re


class ChatbotFunctions:
    """
    A class containing all available functions that the LLM can call,
    with improved fuzzy matching and confidence thresholds.
    """

    # Configuration
    MIN_CONFIDENCE_THRESHOLD = 70  # Minimum confidence for fuzzy matching

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
        "get_all_cuisines": {
            "name": "get_all_cuisines",
            "description": "Get all cuisine types available at various FoodieSpot joints across Bengaluru. Use this function to show the user all available cuisine types.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        "get_area_by_cuisine": {
            "name": "get_area_by_cuisine",
            "description": "Get all the areas serving a specific type of cuisine. Use this function to fetch all FoodieSpot locations serving a specific type of cuisine that the user is interested in.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cuisine": {
                        "type": "string",
                        "description": "The particular cuisine type the user is interested in having. e.g.: 'South Indian', 'Mediterranean' etc.",
                    }
                },
                "required": ["cuisine"],
            },
        },
        "recommend_restaurants": {
            "name": "recommend_restaurants",
            "description": "Recommend restaurants using various filters. Use this function to recommend restaurants to users based on the filters acquired so far.",
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": "The area in Bengaluru the user wants to lookup the FoodieSpot joint in. e.g.: 'Koramangala', 'Whitefield' etc.",
                    },
                    "cuisine": {
                        "type": "string",
                        "description": "The type of food the user wishes to have during their dine-out. e.g.: 'South Indian', 'Mediterranean' etc.",
                    },
                },
                "required": ["area"],
            },
        },
        "make_reservation": {
            "name": "make_reservation",
            "description": "Reserve a table for the user at a specific FoodieSpot location. Use this function to allow the user to make a reservation at their desired FoodieSpot joint.",
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant": {
                        "type": "string",
                        "description": "The name of the FoodieSpot joint the user wishes to make a reservation at. Ensure it is the one the user wishes to dine at. e.g. 'FoodieSpot - Marathahalli', 'FoodieSpot - Rajajinagar' etc.",
                    },
                    "name": {
                        "type": "string",
                        "description": "The full name of the user.",
                    },
                    "phone_number": {
                        "type": "string",
                        "description": "The phone number of the user.",
                    },
                    "headcount": {
                        "type": "number",
                        "description": "The number of people who will be dining.",
                    },
                    "time_slot": {
                        "type": "object",
                        "description": "The time slot for which the user is making a booking at the restaurant. Must be 24-hour time only. If unclear, ask the user to specify AM or PM.",
                        "properties": {
                            "hour": {
                                "type": "number",
                                "description": "The hour at which the user will be arriving at the restaurant. Use 24-hour time only. If unclear, ask the user to specify AM or PM.",
                            },
                            "minute": {
                                "type": "number",
                                "description": "The minute at which the user will be arriving at the restaurant. Use 24-hour time only. If unclear, ask the user to specify AM or PM.",
                            },
                        },
                        "required": ["hour", "minute"],
                    },
                },
                "required": [
                    "restaurant",
                    "name",
                    "phone_number",
                    "headcount",
                    "time_slot",
                ],
            },
        },
    }

    __restaurants_data = json.load(open("./data/restaurants_data.json", "r"))
    __reservations_data = json.load(open("./data/reservations_data.json", "r"))

    @classmethod
    def get_matching_locations(cls, area: str, top_n: int = 5) -> str:
        """
        Get matching FoodieSpot restaurant locations with confidence thresholds.
        Only returns matches above the minimum confidence threshold.
        """
        try:
            # Create list of tuples (area, place_dict)
            area_place_pairs = [
                (place["location"]["area"], place) for place in cls.__restaurants_data
            ]

            # Extract areas list for matching
            areas = [pair[0] for pair in area_place_pairs]

            # Get top N matches using RapidFuzz with confidence threshold
            top_matches = process.extract(
                area, areas, scorer=fuzz.partial_ratio, limit=top_n
            )

            # Filter matches by confidence threshold
            filtered_matches = [
                (matched_area, score, idx)
                for matched_area, score, idx in top_matches
                if score >= cls.MIN_CONFIDENCE_THRESHOLD
            ]

            if not filtered_matches:
                return f"No FoodieSpot locations found matching '{area}'. Please try a different area name or check spelling."

            results = []
            for matched_area, score, idx in filtered_matches:
                place = area_place_pairs[idx][1]
                results.append({"area": place["location"]["area"], "confidence": score})

            area_names = [result["area"] for result in results]

            return json.dumps(
                {
                    "status": "success",
                    "message": f"Found {len(results)} matching FoodieSpot locations",
                    "locations": area_names,
                    "instruction": "Show these locations as numbered options and ask user to select one",
                }
            )

        except Exception as e:
            return json.dumps(
                {"status": "error", "message": f"Error finding locations: {str(e)}"}
            )

    @classmethod
    def get_cuisine_by_area(cls, area: str) -> str:
        """
        Returns cuisines available in the specified area with improved matching.
        """
        try:
            cuisines = set()
            area_found = False

            for restaurant in cls.__restaurants_data:
                rest_location = restaurant.get("location", {}).get("area", "").lower()
                if area.lower() == rest_location:
                    area_found = True
                    cuisine_field = restaurant.get("cuisine", [])
                    if isinstance(cuisine_field, list):
                        cuisines.update(cuisine_field)
                    elif isinstance(cuisine_field, str):
                        cuisines.add(cuisine_field)

            if not area_found:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Area '{area}' not found. Please use get_matching_locations first to confirm the area.",
                    }
                )

            if not cuisines:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"No cuisines found for area '{area}'. Please try another location.",
                    }
                )

            return json.dumps(
                {
                    "status": "success",
                    "area": area,
                    "cuisines": sorted(list(cuisines)),
                    "instruction": "Show these cuisines as numbered options for user selection",
                }
            )

        except Exception as e:
            return json.dumps(
                {"status": "error", "message": f"Error getting cuisines: {str(e)}"}
            )

    @classmethod
    def get_all_cuisines(cls) -> str:
        """
        Returns all available cuisine types across all FoodieSpot locations.
        """
        try:
            cuisine_set = set()
            for restaurant in cls.__restaurants_data:
                cuisine_field = restaurant.get("cuisine", [])
                if isinstance(cuisine_field, list):
                    cuisine_set.update(cuisine_field)
                elif isinstance(cuisine_field, str):
                    cuisine_set.add(cuisine_field)

            if not cuisine_set:
                return json.dumps(
                    {"status": "error", "message": "No cuisines found in database"}
                )

            return json.dumps(
                {
                    "status": "success",
                    "cuisines": sorted(list(cuisine_set)),
                    "total_count": len(cuisine_set),
                    "instruction": "Show these cuisines as numbered options for user selection",
                }
            )

        except Exception as e:
            return json.dumps(
                {"status": "error", "message": f"Error getting all cuisines: {str(e)}"}
            )

    @classmethod
    def get_area_by_cuisine(cls, cuisine: str) -> str:
        """
        Returns areas serving the specified cuisine with improved matching.
        """
        try:
            areas = set()
            cuisine_found = False

            for restaurant in cls.__restaurants_data:
                cuisine_field = restaurant.get("cuisine", [])
                restaurant_cuisines = []

                if isinstance(cuisine_field, list):
                    restaurant_cuisines = [c.lower().strip() for c in cuisine_field]
                elif isinstance(cuisine_field, str):
                    restaurant_cuisines = [cuisine_field.lower().strip()]

                if cuisine.lower().strip() in restaurant_cuisines:
                    cuisine_found = True
                    area = restaurant.get("location", {}).get("area")
                    if area:
                        areas.add(area)

            if not cuisine_found:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Cuisine '{cuisine}' not found. Use get_all_cuisines to see available options.",
                    }
                )

            if not areas:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"No areas found serving '{cuisine}' cuisine.",
                    }
                )

            return json.dumps(
                {
                    "status": "success",
                    "cuisine": cuisine,
                    "areas": sorted(list(areas)),
                    "instruction": "Show these areas as numbered options for user selection",
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Error finding areas for cuisine: {str(e)}",
                }
            )

    @classmethod
    def recommend_restaurants(cls, area: str, cuisine: str = None) -> str:
        """
        Recommend restaurants with improved filtering and error handling.
        """
        try:
            recommendations = []
            area_found = False

            for restaurant in cls.__restaurants_data:
                rest_location = restaurant.get("location", {}).get("area", "").lower()

                # Check if area matches
                if area.lower() != rest_location:
                    continue

                area_found = True

                # If cuisine is specified, filter by cuisine
                if cuisine:
                    rest_cuisines = restaurant.get("cuisine", [])

                    if isinstance(rest_cuisines, str):
                        cuisine_list = [rest_cuisines.lower().strip()]
                    elif isinstance(rest_cuisines, list):
                        cuisine_list = [c.lower().strip() for c in rest_cuisines]
                    else:
                        continue

                    if cuisine.lower().strip() not in cuisine_list:
                        continue  # Skip if cuisine doesn't match

                # Add restaurant to recommendations
                recommendations.append(restaurant)

            if not area_found:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Area '{area}' not found. Please use get_matching_locations first.",
                    }
                )

            if not recommendations:
                cuisine_msg = f" serving '{cuisine}' cuisine" if cuisine else ""
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"No restaurants found in '{area}'{cuisine_msg}.",
                    }
                )

            return json.dumps(
                {
                    "status": "success",
                    "area": area,
                    "cuisine": cuisine,
                    "restaurants": recommendations,
                    "count": len(recommendations),
                    "instruction": "Present these restaurants to the user and ask if they want to make a reservation",
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Error getting restaurant recommendations: {str(e)}",
                }
            )

    @classmethod
    def make_reservation(
        cls,
        restaurant,
        name,
        phone_number,
        headcount,
        time_slot,
    ):
        # 1. Validate restaurant
        matching_restaurant = next(
            (r for r in cls.__restaurants_data if r["name"] == restaurant), None
        )
        if not matching_restaurant:
            return json.dumps(
                {
                    "success": False,
                    "error": "Invalid restaurant name. Please choose a valid FoodieSpot location.",
                }
            )

        # 2. Validate phone number
        if not (
            isinstance(phone_number, str) and re.fullmatch(r"\d{10}", phone_number)
        ):
            return json.dumps(
                {
                    "success": False,
                    "error": "Invalid phone number. It must be a 10-digit number.",
                }
            )

        # 3. Validate headcount
        if not isinstance(headcount, int) or headcount <= 0:
            return json.dumps(
                {
                    "success": False,
                    "error": "Invalid headcount. It must be a positive number.",
                }
            )
        if headcount > matching_restaurant["seating_capacity"]:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Requested headcount exceeds seating capacity of {matching_restaurant['seating_capacity']}.",
                }
            )

        # 4. Validate time_slot
        if not isinstance(time_slot, dict):
            return json.dumps(
                {"success": False, "error": "Missing or invalid time_slot."}
            )

        hour = time_slot.get("hour")
        minute = time_slot.get("minute")

        if not (isinstance(hour, int) and 0 <= hour <= 23):
            return json.dumps(
                {
                    "success": False,
                    "error": "Invalid 'hour' in time_slot. Must be between 0 and 23.",
                }
            )
        if not (isinstance(minute, int) and 0 <= minute <= 59):
            return json.dumps(
                {
                    "success": False,
                    "error": "Invalid 'minute' in time_slot. Must be between 0 and 59.",
                }
            )

        # Passed all validations
        new_reservation = {
            "restaurant": restaurant,
            "name": name,
            "phone_number": phone_number,
            "headcount": headcount,
            "time_slot": {"hour": hour, "minute": minute},
        }

        cls.__reservations_data.append(new_reservation)
        with open("./data/reservations_data.json", "w") as file:
            json.dump(cls.__reservations_data, file, indent=2)

        return json.dumps(
            {
                "success": True,
                "message": "Reservation successful!",
                "reservation": new_reservation,
            },
            indent=2,
        )

    @classmethod
    def get_descriptions(cls):
        """Return function descriptions for tool calling"""
        return cls.__descriptions_all.values()
