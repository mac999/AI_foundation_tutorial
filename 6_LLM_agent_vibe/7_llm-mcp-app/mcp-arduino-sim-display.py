import pygame
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import threading
import uvicorn

# Global variable to store PIN states
pin_states = {i: 0 for i in range(14)}  # Simulating 14 digital pins (0-13)

# FastAPI app
app = FastAPI()

@app.get("/read_pin/{pin}")
async def read_pin(pin: int):
    """
    API endpoint to read the value of a specific PIN.
    Args:
        pin (int): The PIN number to read.
    Returns:
        JSONResponse: The current value of the PIN (0 or 1).
    """
    if pin not in pin_states:
        return JSONResponse(content={"error": "Invalid PIN number"}, status_code=400)
    return {"pin": pin, "value": pin_states[pin]}

@app.post("/write_pin/{pin}/{value}")
async def write_pin(pin: int, value: int):
    """
    API endpoint to write a value to a specific PIN.
    Args:
        pin (int): The PIN number to write to.
        value (int): The value to set (0 or 1).
    Returns:
        JSONResponse: Confirmation of the operation.
    """
    if pin not in pin_states:
        return JSONResponse(content={"error": "Invalid PIN number"}, status_code=400)
    if value not in (0, 1):
        return JSONResponse(content={"error": "Value must be 0 or 1"}, status_code=400)
    pin_states[pin] = value
    return {"message": f"PIN {pin} set to {value}"}

# Pygame UI to display PIN states
def display_ui():
    """
    Display a graphical UI using Pygame to visualize the states of the Arduino PINs.
    """
    pygame.init()
    screen = pygame.display.set_mode((800, 400))
    pygame.display.set_caption("Arduino Uno - PIN States")
    clock = pygame.time.Clock()

    # Define colors
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)

    running = True
    while running:
        screen.fill(BLACK)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Render PIN states as LEDs
        x, y = 50, 50
        radius = 20
        spacing = 100
        for pin, state in pin_states.items():
            color = GREEN if state == 1 else RED
            pygame.draw.circle(screen, color, (x, y), radius)
            font = pygame.font.Font(None, 24)
            text = font.render(f"PIN {pin}", True, WHITE)
            screen.blit(text, (x - radius, y + radius + 10))
            x += spacing
            if x > 750:  # Move to the next row if too wide
                x = 50
                y += 100

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

# Run FastAPI server in a separate thread
def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    # Start the FastAPI server in a separate thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Start the Pygame UI
    display_ui()