import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import imageio
import time
import random
import math
from custom_env import CropStage, Action, FarmEnv

class FarmRenderer:
    def __init__(self, env):
        self.env = env
        self.grid_size = env.grid_size
        self.cell_size = 1.0
        self.window_size = 800
        self.aspect_ratio = 1.0

        # Animation
        self.rotation_angle = 0
        self.agent_bounce = 0

        # GIF Recording
        self.frames = []
        self.recording = True
        self.frame_count = 0
        self.max_frames = 150

        # Initialize OpenGL
        glutInit()
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        glutInitWindowSize(self.window_size, self.window_size)
        self.window = glutCreateWindow(b"Smart Farm Yield Optimization")

        glutDisplayFunc(self.render_scene)
        glutIdleFunc(self.update)

        # Enable features
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)

        # Set up lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [5, 5, 10, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])

        # Set up the view
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, self.aspect_ratio, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

        # Enhanced color palette
        self.colors = {
            CropStage.BARE: [0.6, 0.4, 0.2],      # Brown soil
            CropStage.PLANTED: [0.8, 0.9, 0.6],   # Light green
            CropStage.GROWING: [0.4, 0.8, 0.3],   # Medium green
            CropStage.READY: [0.2, 0.6, 0.1],     # Dark green
            CropStage.DAMAGED: [0.8, 0.4, 0.2],   # Orange-brown
            'moisture': [0.3, 0.6, 1.0],          # Blue
            'nutrient': [0.9, 0.7, 0.3],          # Golden
            'pest': [0.9, 0.1, 0.1],              # Red
            'agent': [0.2, 0.2, 0.9],             # Blue
            'action_colors': {
                Action.PLANT.value: [0.0, 1.0, 0.0],
                Action.IRRIGATE.value: [0.0, 0.5, 1.0],
                Action.FERTILIZE.value: [0.8, 0.5, 0.2],
                Action.HARVEST.value: [1.0, 1.0, 0.0]
            }
        }

        # Label mappings for different cell types
        self.cell_labels = {
            CropStage.BARE: "BARE",
            CropStage.PLANTED: "PLANTED", 
            CropStage.GROWING: "GROWING",
            CropStage.READY: "READY",
            CropStage.DAMAGED: "DAMAGED"
        }

        # Action labels - only the main actions
        self.action_labels = {
            Action.PLANT.value: "PLANT",
            Action.IRRIGATE.value: "IRRIGATE",
            Action.FERTILIZE.value: "FERTILIZE",
            Action.HARVEST.value: "HARVEST"
        }

    def render_text(self, x, y, z, text):
        glPushMatrix()
        glTranslatef(x, y, z)
        glScalef(0.001, 0.001, 0.001)
        glDisable(GL_LIGHTING)
        glColor3f(1.0, 1.0, 1.0)
        for char in text:
            glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, ord(char))
        glEnable(GL_LIGHTING)
        glPopMatrix()

    def render_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        camera_x = self.grid_size/2 + 3 * math.cos(self.rotation_angle * 0.01)
        camera_y = -self.grid_size + 2 * math.sin(self.rotation_angle * 0.01)
        camera_z = self.grid_size * 1.2
        gluLookAt(camera_x, camera_y, camera_z,
                 self.grid_size/2, self.grid_size/2, 0,
                 0, 0, 1)
        self.draw_farm_grid()
        self.draw_agent()
        self.draw_cell_labels()
        self.draw_agent_action_label()
        self.draw_floating_info()
        glutSwapBuffers()
        if self.recording and self.frame_count < self.max_frames:
            self.capture_frame()

    def draw_cell_labels(self):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                crop_state = CropStage(self.env.grid[x, y])
                label = self.cell_labels[crop_state]
                label_x = x
                label_y = y
                label_z = 0.8
                glPushMatrix()
                glTranslatef(label_x, label_y, label_z - 0.1)
                glDisable(GL_LIGHTING)
                glColor4f(0.0, 0.0, 0.0, 0.7)
                glScalef(0.8, 0.2, 0.05)
                glutSolidCube(1.0)
                glEnable(GL_LIGHTING)
                glPopMatrix()
                self.render_text(label_x - 0.3, label_y, label_z, label)

    def draw_agent_action_label(self):
        if hasattr(self.env, 'last_cell') and hasattr(self.env, 'last_action'):
            cell_x, cell_y = self.env.last_cell
            last_action = self.env.last_action
            if last_action is not None:
                action_label = self.action_labels.get(last_action, "Unknown Action")
                glPushMatrix()
                glTranslatef(cell_x, cell_y, 1.2)
                glDisable(GL_LIGHTING)
                glColor4f(0.2, 0.2, 0.8, 0.8)
                glScalef(1.2, 0.3, 0.1)
                glutSolidCube(1.0)
                glEnable(GL_LIGHTING)
                glPopMatrix()
                glDisable(GL_LIGHTING)
                glColor3f(1.0, 1.0, 0.0)
                self.render_text(cell_x - 0.4, cell_y, 1.2, action_label)
                glEnable(GL_LIGHTING)

    def draw_farm_grid(self):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.draw_enhanced_cell(x, y)

    def draw_enhanced_cell(self, x, y):
        glPushMatrix()
        glTranslatef(x, y, 0)
        crop_state = CropStage(self.env.grid[x, y])
        moisture = self.env.moisture[x, y]
        nutrients = self.env.nutrients[x, y]
        has_pest = self.env.pests[x, y]
        height = 0.1 + (moisture + nutrients) * 0.2
        glPushMatrix()
        glScalef(0.95, 0.95, height)
        glColor3fv(self.colors[crop_state])
        glutSolidCube(1.0)
        glPopMatrix()
        if crop_state != CropStage.BARE:
            self.draw_crop_visual(crop_state, moisture, nutrients)
        if moisture > 0.1:
            glPushMatrix()
            glTranslatef(-0.35, -0.35, 0.1)
            glColor3fv(self.colors['moisture'])
            glScalef(moisture * 0.3, 0.1, moisture * 0.5)
            glutSolidCube(1.0)
            glPopMatrix()
        if nutrients > 0.1:
            glPushMatrix()
            glTranslatef(0.35, -0.35, 0.1)
            glColor3fv(self.colors['nutrient'])
            glScalef(nutrients * 0.3, nutrients * 0.3, nutrients * 0.4)
            glutSolidSphere(0.5, 8, 8)
            glPopMatrix()
        if has_pest:
            glPushMatrix()
            glTranslatef(0, 0, 0.3)
            glColor3fv(self.colors['pest'])
            glRotatef(self.rotation_angle * 2, 0, 0, 1)
            glutSolidCone(0.1, 0.4, 6, 6)
            glPopMatrix()
        glColor3f(0.2, 0.2, 0.2)
        glPushMatrix()
        glTranslatef(0, 0, 0.05)
        glutWireCube(0.98)
        glPopMatrix()
        glPopMatrix()

    def draw_crop_visual(self, crop_state, moisture, nutrients):
        growth_factor = (moisture + nutrients) / 2
        if crop_state == CropStage.PLANTED:
            glColor3f(0.6, 0.9, 0.4)
            for i in range(3):
                glPushMatrix()
                glTranslatef((i-1)*0.2, (i-1)*0.15, 0.15)
                glutSolidSphere(0.05, 6, 6)
                glPopMatrix()
        elif crop_state == CropStage.GROWING:
            glColor3f(0.3, 0.7, 0.2)
            for i in range(4):
                glPushMatrix()
                glTranslatef((i-1.5)*0.15, (i-2)*0.1, 0.1)
                glScalef(0.05, 0.05, growth_factor * 0.6)
                glutSolidCube(1.0)
                glTranslatef(0, 0, 0.8)
                glColor3f(0.4, 0.8, 0.3)
                glutSolidSphere(0.1, 6, 6)
                glPopMatrix()
        elif crop_state == CropStage.READY:
            glColor3f(0.2, 0.5, 0.1)
            for i in range(5):
                glPushMatrix()
                glTranslatef((i-2)*0.15, random.uniform(-0.2, 0.2), 0.1)
                glScalef(0.08, 0.08, growth_factor * 0.8)
                glutSolidCube(1.0)
                glTranslatef(0, 0, 0.9)
                glColor3fv([1.0, 0.8, 0.3])
                glutSolidSphere(0.12, 8, 8)
                glPopMatrix()
        elif crop_state == CropStage.DAMAGED:
            glColor3f(0.6, 0.3, 0.1)
            for i in range(3):
                glPushMatrix()
                glTranslatef((i-1)*0.2, (i-1)*0.1, 0.05)
                glRotatef(random.uniform(-30, 30), 1, 1, 0)
                glScalef(0.06, 0.06, 0.3)
                glutSolidCube(1.0)
                glPopMatrix()

    def draw_agent(self):
        # Visualize the last cell acted on
        if hasattr(self.env, 'last_cell'):
            cell_x, cell_y = self.env.last_cell
            glPushMatrix()
            glTranslatef(cell_x, cell_y, 0.5 + 0.1 * math.sin(self.agent_bounce * 0.1))
            glColor3fv(self.colors['agent'])
            glutSolidCube(0.4)
            glPushMatrix()
            glTranslatef(0, 0, 0.3)
            glScalef(0.6, 0.6, 0.6)
            glColor3f(0.1, 0.1, 0.7)
            glutSolidCube(0.4)
            glPopMatrix()
            last_action = getattr(self.env, 'last_action', None)
            if last_action is not None:
                glPushMatrix()
                glTranslatef(0, 0, 0.5)
                glRotatef(self.rotation_angle * 3, 0, 0, 1)
                glColor3fv(self.colors['action_colors'].get(last_action, [1.0, 1.0, 1.0]))
                glScalef(1.0, 1.0, 0.1)
                glutSolidCube(0.3)
                glPopMatrix()
            glPopMatrix()

    def draw_floating_info(self):
        glPushMatrix()
        glTranslatef(self.grid_size/2, self.grid_size + 1, 2)
        glColor4f(0.1, 0.1, 0.1, 0.8)
        glScalef(2, 0.1, 0.3)
        glutSolidCube(1.0)
        glPopMatrix()
        stats_text = f"Step: {self.env.steps} | Yield: {self.env.total_yield:.1f} | Harvests: {self.env.harvests_count}"
        self.render_text(self.grid_size/2 - 1, self.grid_size + 1, 2.1, stats_text)

    def update(self):
        if self.frame_count >= self.max_frames and self.recording:
            self.save_gif()
            self.recording = False
            glutLeaveMainLoop()
            return
        self.rotation_angle += 1
        self.agent_bounce += 1
        if hasattr(self.env, 'action_space'):
            action = self.env.action_space.sample()
            self.env.step(action)
            if self.env.steps >= 200:
                self.env.reset()
        time.sleep(0.05)
        glutPostRedisplay()

    def capture_frame(self):
        glReadBuffer(GL_FRONT)
        buffer = glReadPixels(0, 0, self.window_size, self.window_size, 
                            GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(buffer, dtype=np.uint8).reshape(
            self.window_size, self.window_size, 3)
        self.frames.append(np.flipud(image))
        self.frame_count += 1

    def save_gif(self, filename="enhanced_farm_simulation.gif"):
        if len(self.frames) > 0:
            selected_frames = self.frames[::2]
            imageio.mimsave(filename, selected_frames, fps=8, loop=0)
            print(f"Saved enhanced simulation as {filename} with {len(selected_frames)} frames")
        else:
            print("No frames captured to save")

    def run_simulation(self):
        print("Starting Enhanced Farm Yield Optimization Simulation")
        print("Features: 3D Crops, Animated Agent, Dynamic Camera, Environmental Indicators, Cell Labels")
        print(f"Recording {self.max_frames} frames to GIF...")
        glutMainLoop()

    def close(self):
        if self.recording and len(self.frames) > 0:
            self.save_gif()
        glutDestroyWindow(self.window)

def main():
    env = FarmEnv()
    renderer = FarmRenderer(env)
    try:
        renderer.run_simulation()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        renderer.close()

if __name__ == "__main__":
    main()