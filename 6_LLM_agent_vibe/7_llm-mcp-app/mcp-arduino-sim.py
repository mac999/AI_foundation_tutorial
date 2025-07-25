import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arduino-sim")

# Simulated PIN states
pin_states = {}

# Write to a PIN in Arduino
@mcp.tool()
def write_pin(pin: int, value: int) -> str:
    """
    Write a value to a specific PIN.
    Args:
        pin (int): The PIN number.
        value (int): The value to write (0 or 1).
    Returns:
        str: Confirmation message.
    """
    if value not in (0, 1):
        raise ValueError("Value must be 0 or 1")
    pin_states[pin] = value
    print(f"Writing value {value} to PIN {pin}")

    # Send request to the display server
    try:
        response = requests.post(f"http://127.0.0.1:8000/write_pin/{pin}/{value}")
        if response.status_code != 200:
            print(f"Failed to update display server: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to display server: {e}")

    return f"PIN {pin} set to {value}"

# Read from a PIN in Arduino
@mcp.tool()
def read_pin(pin: int) -> int:
    """
    Read the value of a specific PIN in Arduino.
    Args:
        pin (int): The PIN number.
    Returns:
        int: The value of the PIN (0 or 1).
    """
    value = pin_states.get(pin, 0)
    print(f"Reading value {value} from PIN {pin}")

    # Send request to the display server
    try:
        response = requests.get(f"http://127.0.0.1:8000/read_pin/{pin}")
        if response.status_code == 200:
            value = response.json().get("value", 0)
        else:
            print(f"Failed to fetch from display server: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to display server: {e}")

    return value

# Get all PIN states in Arduino
@mcp.resource("pins://all")
def get_all_pins() -> dict:
    """
    Get the states of all PINs.
    Returns:
        dict: A dictionary of PIN states.
    """
    return pin_states

def test():
	write_pin(1, 1)
	write_pin(2, 1)
	write_pin(3, 1)
	write_pin(1, 0)
	pin = read_pin(1)
	print(pin)
	pins = get_all_pins()
	print(pins)

# test()

if __name__ == "__main__":
    mcp.run()