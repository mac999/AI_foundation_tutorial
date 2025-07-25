from mcp.server.fastmcp import FastMCP

mcp = FastMCP("arduino-sim")

# Simulated PIN states
pin_states = {}

# Write to a PIN
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
	print(f"Writing value {value} to PIN {pin}")  # Log the write operation
	return f"PIN {pin} set to {value}"

# Read from a PIN
@mcp.tool()
def read_pin(pin: int) -> int:
	"""
	Read the value of a specific PIN.
	Args:
		pin (int): The PIN number.
	Returns:
		int: The value of the PIN (0 or 1).
	"""
	value = pin_states.get(pin, 0)  # Default to 0 if PIN is not set
	print(f"Reading value {value} from PIN {pin}")  # Log the read operation
	return value

# Get all PIN states
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
	write_pin(2, 0)
	read_pin(1)
	pins = get_all_pins()
	print(pins)

if __name__ == "__main__":
	mcp.run()