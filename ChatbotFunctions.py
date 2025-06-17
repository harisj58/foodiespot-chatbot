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
        "get_area_by_ambience": {
            "name": "get_area_by_ambience",
            "description": "Get all the areas that have restaurants with a specific ambience. Use this function to fetch all FoodieSpot locations with a specific ambience that the user is interested in.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ambience": {
                        "type": "string",
                        "description": "The particular ambience type the user is interested in. e.g.: 'Trendy', 'Casual', 'Fine Dining' etc.",
                    }
                },
                "required": ["ambience"],
            },
        },
        "get_ambience_by_area": {
            "name": "get_ambience_by_area",
            "description": "Get the types of ambience available at restaurants in a particular area in Bengaluru. Use this function to show ambience options for a specific area. Make sure the area you are looking for is exactly the same as the one you get from `get_matching_locations`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "description": "The area in Bengaluru the user wants to lookup the FoodieSpot joint ambience in. e.g.: 'Koramangala', 'Whitefield' etc.",
                    }
                },
                "required": ["area"],
            },
        },
        "get_all_ambiences": {
            "name": "get_all_ambiences",
            "description": "Get all ambience types available at various FoodieSpot joints across Bengaluru. Use this function to show the user all available ambience types.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
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
                    "ambience": {
                        "type": "string",
                        "description": "The type of ambience the user prefers for their dining experience. e.g.: 'Trendy', 'Casual', 'Fine Dining' etc.",
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
    def get_area_by_ambience(cls, ambience: str) -> str:
        """
        Returns areas that have restaurants with the specified ambience using fuzzy matching.
        """
        try:
            # Get all unique ambiences from restaurants
            all_ambiences = []
            restaurant_ambience_map = {}

            for restaurant in cls.__restaurants_data:
                rest_ambience = restaurant.get("ambience", "")
                if rest_ambience:
                    all_ambiences.append(rest_ambience)
                    if rest_ambience not in restaurant_ambience_map:
                        restaurant_ambience_map[rest_ambience] = []
                    restaurant_ambience_map[rest_ambience].append(restaurant)

            if not all_ambiences:
                return json.dumps(
                    {"status": "error", "message": "No ambience data found in database"}
                )

            # Use fuzzy matching to find the best ambience match
            best_match = process.extractOne(
                ambience, all_ambiences, scorer=fuzz.partial_ratio
            )

            if not best_match or best_match[1] < cls.MIN_CONFIDENCE_THRESHOLD:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Ambience '{ambience}' not found. Use get_all_ambiences to see available options.",
                    }
                )

            matched_ambience = best_match[0]
            areas = set()

            # Get all areas that have restaurants with this ambience
            for restaurant in restaurant_ambience_map[matched_ambience]:
                area = restaurant.get("location", {}).get("area")
                if area:
                    areas.add(area)

            if not areas:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"No areas found with '{matched_ambience}' ambience.",
                    }
                )

            return json.dumps(
                {
                    "status": "success",
                    "ambience": matched_ambience,
                    "areas": sorted(list(areas)),
                    "confidence": best_match[1],
                    "instruction": "Show these areas as numbered options for user selection",
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Error finding areas for ambience: {str(e)}",
                }
            )

    @classmethod
    def get_ambience_by_area(cls, area: str) -> str:
        """
        Returns ambiences available in the specified area.
        """
        try:
            ambiences = set()
            area_found = False

            for restaurant in cls.__restaurants_data:
                rest_location = restaurant.get("location", {}).get("area", "").lower()
                if area.lower() == rest_location:
                    area_found = True
                    ambience = restaurant.get("ambience", "")
                    if ambience:
                        ambiences.add(ambience)

            if not area_found:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Area '{area}' not found. Please use get_matching_locations first to confirm the area.",
                    }
                )

            if not ambiences:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"No ambience data found for area '{area}'. Please try another location.",
                    }
                )

            return json.dumps(
                {
                    "status": "success",
                    "area": area,
                    "ambiences": sorted(list(ambiences)),
                    "instruction": "Show these ambiences as numbered options for user selection",
                }
            )

        except Exception as e:
            return json.dumps(
                {"status": "error", "message": f"Error getting ambiences: {str(e)}"}
            )

    @classmethod
    def get_all_ambiences(cls) -> str:
        """
        Returns all available ambience types across all FoodieSpot locations.
        """
        try:
            ambience_set = set()
            for restaurant in cls.__restaurants_data:
                ambience = restaurant.get("ambience", "")
                if ambience:
                    ambience_set.add(ambience)

            if not ambience_set:
                return json.dumps(
                    {"status": "error", "message": "No ambience data found in database"}
                )

            return json.dumps(
                {
                    "status": "success",
                    "ambiences": sorted(list(ambience_set)),
                    "total_count": len(ambience_set),
                    "instruction": "Show these ambiences as numbered options for user selection",
                }
            )

        except Exception as e:
            return json.dumps(
                {"status": "error", "message": f"Error getting all ambiences: {str(e)}"}
            )

    @classmethod
    def recommend_restaurants(
        cls, area: str, cuisine: str = None, ambience: str = None
    ) -> str:
        """
        Recommend restaurants with improved filtering and error handling.
        Now supports filtering by area, cuisine, and ambience.
        """
        try:
            recommendations = []
            area_found = False
            matched_ambience = None

            # If ambience is provided, find the best fuzzy match first
            if ambience:
                all_ambiences = [
                    r.get("ambience", "")
                    for r in cls.__restaurants_data
                    if r.get("ambience")
                ]
                if all_ambiences:
                    best_ambience_match = process.extractOne(
                        ambience, all_ambiences, scorer=fuzz.partial_ratio
                    )
                    if (
                        best_ambience_match
                        and best_ambience_match[1] >= cls.MIN_CONFIDENCE_THRESHOLD
                    ):
                        matched_ambience = best_ambience_match[0]

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

                # If ambience is specified, filter by ambience
                if ambience and matched_ambience:
                    rest_ambience = restaurant.get("ambience", "")
                    if (
                        rest_ambience.lower().strip()
                        != matched_ambience.lower().strip()
                    ):
                        continue  # Skip if ambience doesn't match

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
                filters = []
                if cuisine:
                    filters.append(f"serving '{cuisine}' cuisine")
                if ambience and matched_ambience:
                    filters.append(f"with '{matched_ambience}' ambience")
                elif ambience and not matched_ambience:
                    filters.append(f"with '{ambience}' ambience (no close match found)")

                filter_msg = " " + " and ".join(filters) if filters else ""

                return json.dumps(
                    {
                        "status": "error",
                        "message": f"No restaurants found in '{area}'{filter_msg}.",
                    }
                )

            result = {
                "status": "success",
                "area": area,
                "restaurants": recommendations,
                "count": len(recommendations),
                "instruction": "Present these restaurants to the user and ask if they want to make a reservation",
            }

            if cuisine:
                result["cuisine"] = cuisine
            if ambience and matched_ambience:
                result["ambience"] = matched_ambience

            return json.dumps(result)

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
        if name.lower() == "user" or phone_number.lower() == "user":
            return json.dumps(
                {
                    "success": False,
                    "error": "Need the following details to make reservation: name, phone number, head count and time slot. Ask the user for these details.",
                }
            )

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
