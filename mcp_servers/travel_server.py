
from pathlib import Path

from mcp.server import MCPServer

# Create the MCP server
mcp = MCPServer("smartagentic-travel")

# Locate the existing city documents
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CITIES_DIR = PROJECT_ROOT / "data" / "cities"


@mcp.tool()
def list_supported_cities() -> list[str]:
    """List the cities available in the local travel knowledge base."""
    if not CITIES_DIR.exists():
        return []

    return sorted(
        path.stem.replace("_", " ").title()
        for path in CITIES_DIR.glob("*.txt")
        if path.is_file()
    )


@mcp.tool()
def get_city_info(city: str) -> str:
    """Retrieve the local travel knowledge document for a city."""
    normalized_city = city.strip().lower().replace(" ", "_")

    # Accept only a city document that actually exists
    if not normalized_city or not normalized_city.replace("_", "").isalpha():
        return "Invalid city name. Please provide a city using letters only."

    city_file = CITIES_DIR / f"{normalized_city}.txt"

    if not city_file.is_file():
        available = list_supported_cities()
        return (
            f"No local travel document found for '{city}'. "
            f"Available cities: {', '.join(available) or 'none'}."
        )

    return city_file.read_text(encoding="utf-8")


if __name__ == "__main__":
    mcp.run()
