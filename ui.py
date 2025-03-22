import pygame
import numpy as np
from pygame import gfxdraw

class DrawingCanvas:
    def __init__(self, x, y, width, height, line_width=2):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.line_width = line_width
        self.surface = pygame.Surface((width, height))
        self.surface.fill((255, 255, 255))  # White background
        self.drawing = False
        self.last_pos = None
        self.pixels = np.zeros((height, width), dtype=np.uint8)
        self.clear()
        
    def clear(self):
        self.surface.fill((255, 255, 255))
        self.pixels = np.zeros((self.height, self.width), dtype=np.uint8)
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            x, y = event.pos
            if self.is_inside(x, y):
                self.drawing = True
                self.last_pos = (x - self.x, y - self.y)
                self.draw_point(self.last_pos)
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.drawing = False
            self.last_pos = None
            return True
        elif event.type == pygame.MOUSEMOTION and self.drawing:
            x, y = event.pos
            if self.is_inside(x, y):
                current_pos = (x - self.x, y - self.y)
                if self.last_pos:
                    self.draw_line(self.last_pos, current_pos)
                self.last_pos = current_pos
                return True
        return False
    
    def is_inside(self, x, y):
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def draw_point(self, pos):
        x, y = pos
        pygame.draw.circle(self.surface, (0, 0, 0), (x, y), self.line_width)
        # Update pixel array
        for dx in range(-self.line_width, self.line_width + 1):
            for dy in range(-self.line_width, self.line_width + 1):
                if dx*dx + dy*dy <= self.line_width*self.line_width:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        self.pixels[ny, nx] = 255
    
    def draw_line(self, start, end):
        pygame.draw.line(self.surface, (0, 0, 0), start, end, self.line_width * 2)
        # Update pixel array
        x0, y0 = start
        x1, y1 = end
        # Bresenham's line algorithm to update pixels
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            # Draw a circle at each point along the line
            for rdx in range(-self.line_width, self.line_width + 1):
                for rdy in range(-self.line_width, self.line_width + 1):
                    if rdx*rdx + rdy*rdy <= self.line_width*self.line_width:
                        nx, ny = x0 + rdx, y0 + rdy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            self.pixels[ny, nx] = 255
            
            if x0 == x1 and y0 == y1:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
    
    def get_pixels(self):
        return self.pixels
    
    def draw(self, screen):
        screen.blit(self.surface, (self.x, self.y))
        # Draw border
        pygame.draw.rect(screen, (0, 0, 0), 
                         (self.x, self.y, self.width, self.height), 2)

class Button:
    def __init__(self, x, y, width, height, text, action=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.action = action
        self.color = (200, 200, 200)
        self.hover_color = (150, 150, 150)
        self.text_color = (0, 0, 0)
        self.font = pygame.font.SysFont('Arial', 20)
        self.hovering = False
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            x, y = event.pos
            self.hovering = (self.x <= x <= self.x + self.width and 
                             self.y <= y <= self.y + self.height)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            x, y = event.pos
            if (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height and 
                self.action is not None):
                self.action()
                return True
        return False
    
    def draw(self, screen):
        color = self.hover_color if self.hovering else self.color
        pygame.draw.rect(screen, color, (self.x, self.y, self.width, self.height))
        pygame.draw.rect(screen, (0, 0, 0), (self.x, self.y, self.width, self.height), 2)
        
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=(self.x + self.width // 2, 
                                                 self.y + self.height // 2))
        screen.blit(text_surface, text_rect)

class PredictionDisplay:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.font_large = pygame.font.SysFont('Arial', 48, bold=True)
        self.font_small = pygame.font.SysFont('Arial', 16)
        self.prediction = None
        self.probabilities = None
        
    def update(self, prediction, probabilities):
        self.prediction = prediction
        self.probabilities = probabilities
        
    def draw(self, screen):
        # Draw border
        pygame.draw.rect(screen, (0, 0, 0), 
                         (self.x, self.y, self.width, self.height), 2)
        
        if self.prediction is None:
            # Draw placeholder text
            text = self.font_small.render("Draw a digit", True, (100, 100, 100))
            text_rect = text.get_rect(center=(self.x + self.width // 2, 
                                             self.y + self.height // 2))
            screen.blit(text, text_rect)
            return
        
        # Draw the predicted digit
        text = self.font_large.render(str(self.prediction), True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.x + self.width // 2, 
                                         self.y + 50))
        screen.blit(text, text_rect)
        
        # Draw confidence bars
        if self.probabilities is not None:
            bar_height = 15
            bar_margin = 5
            bar_width = self.width - 40
            
            for i, prob in enumerate(self.probabilities):
                # Draw digit label
                digit_text = self.font_small.render(str(i), True, (0, 0, 0))
                screen.blit(digit_text, (self.x + 10, 
                                        self.y + 100 + i * (bar_height + bar_margin)))
                
                # Draw bar background
                pygame.draw.rect(screen, (220, 220, 220), 
                                (self.x + 30, self.y + 100 + i * (bar_height + bar_margin), 
                                 bar_width, bar_height))
                
                # Draw confidence bar
                width = int(bar_width * prob)
                color = (0, 200, 0) if i == self.prediction else (100, 100, 255)
                pygame.draw.rect(screen, color, 
                                (self.x + 30, self.y + 100 + i * (bar_height + bar_margin), 
                                 width, bar_height))
                
                # Draw percentage
                pct_text = self.font_small.render(f"{prob*100:.1f}%", True, (0, 0, 0))
                screen.blit(pct_text, (self.x + 35 + width, 
                                      self.y + 100 + i * (bar_height + bar_margin)))

class UI:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        pygame.display.set_caption("Digit Recognition")
        
        # Initialize UI components
        self.canvas = None
        self.clear_button = None
        self.prediction_display = None
        
        # Calculate layout and create components
        self.recalculate_layout()
        
        self.clock = pygame.time.Clock()
        self.running = True
        
    def recalculate_layout(self):
        # Calculate layout
        canvas_size = min(self.width // 2, self.height - 100)
        canvas_x = (self.width // 2 - canvas_size) // 2
        canvas_y = (self.height - canvas_size) // 2
        
        # Create or update canvas
        if self.canvas is None:
            self.canvas = DrawingCanvas(canvas_x, canvas_y, canvas_size, canvas_size, line_width=15)
        else:
            # Save current drawing if any
            old_pixels = self.canvas.get_pixels()
            # Create new canvas with updated dimensions
            self.canvas = DrawingCanvas(canvas_x, canvas_y, canvas_size, canvas_size, line_width=15)
            # If there was a drawing, we'd need to scale it to the new canvas size
            # This is complex and not implemented here
        
        # Create or update button
        button_width = 100
        button_height = 40
        button_x = canvas_x + (canvas_size - button_width) // 2
        button_y = canvas_y + canvas_size + 20
        self.clear_button = Button(button_x, button_y, button_width, button_height, 
                                  "Clear", self.canvas.clear)
        
        # Create or update prediction display
        prediction_x = self.width // 2 + canvas_x
        prediction_y = canvas_y
        prediction_width = canvas_size
        prediction_height = canvas_size
        if self.prediction_display is None:
            self.prediction_display = PredictionDisplay(prediction_x, prediction_y, 
                                                      prediction_width, prediction_height)
        else:
            # Save current prediction if any
            prediction = self.prediction_display.prediction
            probabilities = self.prediction_display.probabilities
            # Create new display with updated dimensions
            self.prediction_display = PredictionDisplay(prediction_x, prediction_y, 
                                                      prediction_width, prediction_height)
            # Restore prediction if there was one
            if prediction is not None:
                self.prediction_display.update(prediction, probabilities)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.VIDEORESIZE:
                # Update window size
                self.width, self.height = event.size
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                # Recalculate layout
                self.recalculate_layout()
            elif self.canvas.handle_event(event):
                pass
            elif self.clear_button.handle_event(event):
                pass
    
    def update(self, prediction_callback):
        # Get the canvas pixels and pass to prediction callback
        pixels = self.canvas.get_pixels()
        if np.any(pixels):  # Only predict if there's something drawn
            prediction, probabilities = prediction_callback(pixels)
            self.prediction_display.update(prediction, probabilities)
        else:
            self.prediction_display.update(None, None)
    
    def draw(self):
        self.screen.fill((240, 240, 240))  # Light gray background
        self.canvas.draw(self.screen)
        self.clear_button.draw(self.screen)
        self.prediction_display.draw(self.screen)
        pygame.display.flip()
    
    def run(self, prediction_callback):
        while self.running:
            self.handle_events()
            self.update(prediction_callback)
            self.draw()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()
