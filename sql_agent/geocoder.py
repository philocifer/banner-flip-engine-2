from geopy.geocoders import Nominatim

class Geocoder:
    def __init__(self):
        self.geolocator = Nominatim(
            user_agent="banner-flip-app/1.0",  # Required by Nominatim terms
            timeout=10  # Add timeout for reliability
        )
        self.cache = {}  # Add a cache to store previous geocoding results
    
    def geocode(self, location: str) -> tuple[float, float] | None:
        # Check if we already have this location in the cache
        if location in self.cache:
            return self.cache[location]
        
        try:
            print(f"Geocoding location: {location}")
            result = self.geolocator.geocode(location)
            if result:
                # Store the result in the cache before returning
                coordinates = (result.latitude, result.longitude)
                self.cache[location] = coordinates
                return coordinates

            # Cache negative results too to avoid repeated lookups
            self.cache[location] = (None, None)
            return (None, None)
        except Exception as e:
            print(f"Geocoding error: {str(e)}")
            # Don't cache errors as they might be temporary
            return (None, None)

geocoder = Geocoder()

def geocode_location(location: str) -> tuple[float, float] | None:
    return geocoder.geocode(location)