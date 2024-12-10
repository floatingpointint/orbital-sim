import bpy
import mathutils
import math
import random

import scipy.integrate as integrate
import numpy as np


class Orbital_1s:
    @staticmethod
    def rho_function(n, z, r):
        # ρ = 2 * Zr/n
        return 2 * z * r / n

    @staticmethod
    def radial_wave_function(z, p):
        # R1s = 2 x Z^(3/2) x e^(-ρ/2)
        return 2 * z ** (3 / 2) * math.exp(-p / 2)

    @staticmethod
    def angular_wave_function():
        # Y1s = 1 x (1/4π)^(1/2)
        return 1 * (1 / 4 * math.pi) ** (1 / 2)

    @staticmethod
    def superposition_wave_function(r, y):
        # ψ1s = R1s × Y1s
        return r * y

    @staticmethod
    def electron_density_function(wave_function):
        # = ψ1s ^ 2
        return wave_function**2

    @staticmethod
    def radial_distribution_function(r, w):
        # = 4π * r^2 * ψ1s^2
        return 4 * math.pi * r**2 * w**2

    @staticmethod
    def create_electron_template(name, base_radius, colour=(0.0793493, 0.375979, 1, 1)):
        # colour is in (r, g, b, a) format as a percentage from 0 to 1

        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=base_radius / 10,
        )

        current_name = bpy.context.active_object.name
        sphere_object = bpy.data.objects[current_name]

        bpy.data.materials.new(name)
        material = bpy.data.materials[name]

        material.use_nodes = True

        node_tree = material.node_tree

        # Remove the Principled BSDF node if it exists
        principled_node = node_tree.nodes.get("Principled BSDF")
        if principled_node:
            node_tree.nodes.remove(principled_node)

        # Add a Sheen BSDF node
        sheen_node = node_tree.nodes.new(type="ShaderNodeBsdfSheen")

        # Optionally, set the sheen color
        sheen_node.inputs["Color"].default_value = colour

        # Create a material output node if it doesn't exist
        material_output = node_tree.nodes.get("Material Output")
        if not material_output:
            material_output = node_tree.nodes.new(type="ShaderNodeOutputMaterial")

        # Connect the Sheen BSDF to the material output's 'Surface' input
        node_tree.links.new(
            sheen_node.outputs["BSDF"], material_output.inputs["Surface"]
        )

        sphere_object.data.materials.append(material)

        return sphere_object

    @staticmethod
    def create_orbital_segment(name, outer_radius, count, electron_template):
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=outer_radius,
        )

        current_name = bpy.context.active_object.name
        sphere_object = bpy.data.objects[current_name]

        sphere_object.name = name
        sphere_object.data.name = f"{name}_mesh"

        # Hollow sphere applied inwards
        sphere_object.modifiers.new(f"{name}_solidify", type="SOLIDIFY")
        sphere_object.modifiers[f"{name}_solidify"].thickness = segment_1s_radius_step

        # Particle system setup
        pSystem = sphere_object.modifiers.new(
            f"{name}_particles", type="PARTICLE_SYSTEM"
        )

        pSettings = bpy.data.particles.new(f"{name}_particle_settings")

        pSettings.count = count

        pSettings.frame_start = 0
        pSettings.frame_end = 0
        pSettings.lifetime = 1000

        pSettings.emit_from = "VOLUME"
        pSettings.distribution = "RAND"

        pSettings.use_modifier_stack = True
        pSettings.use_emit_random = True
        pSettings.use_even_distribution = True

        pSettings.physics_type = "NO"

        pSettings.render_type = "OBJECT"
        pSettings.instance_object = electron_template
        pSettings.particle_size = 10

        pSystem.particle_system.settings = pSettings

        # Hides wireframe and object while keeping the particles
        sphere_object.show_instancer_for_render = False
        sphere_object.show_instancer_for_viewport = False

        # Segment rotation around z axis
        z_driver = sphere_object.driver_add("rotation_euler", 2).driver
        z_driver.expression = f"frame / 15 / {outer_radius}"

        return sphere_object

    def __init__(self, quantum_number, charge):
        self.quantum_number = quantum_number
        self.charge = charge

        self.cumulative_area, _ = integrate.quad(
            self.integrand, 0, np.inf, args=(self.quantum_number, self.charge)
        )

        print(f"self.cumulative_area = {self.cumulative_area}")

    def integrand(self, x, n, z):
        p = Orbital_1s.rho_function(n, z, x)

        radial_wave = Orbital_1s.radial_wave_function(z, p)
        angular_wave = Orbital_1s.angular_wave_function()
        wave_function = Orbital_1s.superposition_wave_function(
            radial_wave, angular_wave
        )

        return Orbital_1s.radial_distribution_function(x, wave_function)

    def percent_at_radius(self, inner_radius=0.0, outer_radius=1.0):

        area_in_bounds = integrate.quad(
            self.integrand,
            inner_radius,
            outer_radius,
            args=(self.quantum_number, self.charge),
        )

        return area_in_bounds[0] / self.cumulative_area


class Orbital_2s:
    @staticmethod
    def rho_function(n, z, r):
        # ρ = 2 * Zr/n
        return 2 * z * r / n

    @staticmethod
    def radial_wave_function(z, p):
        # R2s = (1/2√2) x (2-ρ) x Z^(3/2) x e^(-ρ/2)
        return 1 / (2 * math.sqrt(2)) * (2 - p) * z ** (3 / 2) * math.exp(-p / 2)

    @staticmethod
    def angular_wave_function():
        # Y2s = 1 x (1/4π)^(1/2)
        return 1 * (1 / 4 * math.pi) ** (1 / 2)

    @staticmethod
    def superposition_wave_function(r, y):
        # ψ2s = R2s × Y2s
        return r * y

    @staticmethod
    def electron_density_function(wave_function):
        # = ψ2s ^ 2
        return wave_function**2

    @staticmethod
    def radial_distribution_function(r, w):
        # = 4π * r^2 * ψ2s^2
        return 4 * math.pi * r**2 * w**2

    @staticmethod
    def create_electron_template(name, base_radius, colour=(0.0793493, 0.375979, 1, 1)):
        # colour is in (r, g, b, a) format as a percentage from 0 to 1

        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=base_radius / 10,
        )

        current_name = bpy.context.active_object.name
        sphere_object = bpy.data.objects[current_name]

        bpy.data.materials.new(name)
        material = bpy.data.materials[name]

        material.use_nodes = True

        node_tree = material.node_tree

        # Remove the Principled BSDF node if it exists
        principled_node = node_tree.nodes.get("Principled BSDF")
        if principled_node:
            node_tree.nodes.remove(principled_node)

        # Add a Sheen BSDF node
        sheen_node = node_tree.nodes.new(type="ShaderNodeBsdfSheen")

        # Optionally, set the sheen color
        sheen_node.inputs["Color"].default_value = colour

        # Create a material output node if it doesn't exist
        material_output = node_tree.nodes.get("Material Output")
        if not material_output:
            material_output = node_tree.nodes.new(type="ShaderNodeOutputMaterial")

        # Connect the Sheen BSDF to the material output's 'Surface' input
        node_tree.links.new(
            sheen_node.outputs["BSDF"], material_output.inputs["Surface"]
        )

        sphere_object.data.materials.append(material)

        return sphere_object

    @staticmethod
    def create_orbital_segment(name, outer_radius, count, electron_template):
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=outer_radius,
        )

        current_name = bpy.context.active_object.name
        sphere_object = bpy.data.objects[current_name]

        sphere_object.name = name
        sphere_object.data.name = f"{name}_mesh"

        # Hollow sphere applied inwards
        sphere_object.modifiers.new(f"{name}_solidify", type="SOLIDIFY")
        sphere_object.modifiers[f"{name}_solidify"].thickness = segment_1s_radius_step

        # Particle system setup
        pSystem = sphere_object.modifiers.new(
            f"{name}_particles", type="PARTICLE_SYSTEM"
        )

        pSettings = bpy.data.particles.new(f"{name}_particle_settings")

        pSettings.count = count

        pSettings.frame_start = 0
        pSettings.frame_end = 0
        pSettings.lifetime = 1000

        pSettings.emit_from = "VOLUME"
        pSettings.distribution = "RAND"

        pSettings.use_modifier_stack = True
        pSettings.use_emit_random = True
        pSettings.use_even_distribution = True

        pSettings.physics_type = "NO"

        pSettings.render_type = "OBJECT"
        pSettings.instance_object = electron_template
        pSettings.particle_size = 10

        pSystem.particle_system.settings = pSettings

        # Hides wireframe and object while keeping the particles
        sphere_object.show_instancer_for_render = False
        sphere_object.show_instancer_for_viewport = False

        # Segment rotation around z axis
        z_driver = sphere_object.driver_add("rotation_euler", 2).driver
        z_driver.expression = f"frame / 15 / {outer_radius}"

        return sphere_object

    def __init__(self, quantum_number, charge):
        self.quantum_number = quantum_number
        self.charge = charge

        self.cumulative_area, _ = integrate.quad(
            self.integrand, 0, np.inf, args=(self.quantum_number, self.charge)
        )

        print(f"self.cumulative_area = {self.cumulative_area}")

    def integrand(self, x, n, z):
        p = Orbital_2s.rho_function(n, z, x)

        radial_wave = Orbital_2s.radial_wave_function(z, p)
        angular_wave = Orbital_2s.angular_wave_function()
        wave_function = Orbital_2s.superposition_wave_function(
            radial_wave, angular_wave
        )

        return Orbital_2s.radial_distribution_function(x, wave_function)

    def percent_at_radius(self, inner_radius=0.0, outer_radius=1.0):

        area_in_bounds = integrate.quad(
            self.integrand,
            inner_radius,
            outer_radius,
            args=(self.quantum_number, self.charge),
        )

        return area_in_bounds[0] / self.cumulative_area


# Items to retain from a new project
PROTECTED_ITEMS = ["Camera", "Light", "Material"]


for obj in bpy.data.objects:
    if obj.name not in PROTECTED_ITEMS:
        bpy.data.objects.remove(obj, do_unlink=True)

for material in bpy.data.materials:
    if material.name not in PROTECTED_ITEMS:
        bpy.data.materials.remove(material, do_unlink=True)


bpy.ops.outliner.orphans_purge()


hydrogenAtom = Orbital_1s(1, 1)
lithiumAtom = Orbital_2s(2, 1)

s1_template = Orbital_1s.create_electron_template("Orbital.1s.Template", 0.1)
s2_template = Orbital_2s.create_electron_template(
    "Orbital.2s.Template", 0.1, (0.2, 0.1, 0.5, 1)
)

segment_1s_radius_range = 10.0
segment_1s_radius_step = 0.1

# Index is not set to 0 because thickness is applied inward
segment_radius_index = segment_1s_radius_step


maximum_electron_count = 100
while segment_radius_index <= segment_1s_radius_range:

    inner_radius = segment_radius_index - segment_1s_radius_step
    outer_radius = segment_radius_index

    percentage = hydrogenAtom.percent_at_radius(inner_radius, outer_radius)

    electron_count = round(percentage * maximum_electron_count)

    Orbital_1s.create_orbital_segment(
        f"Electron.1s.{segment_radius_index}", outer_radius, electron_count, s1_template
    )

    segment_radius_index = round(segment_radius_index + segment_1s_radius_step, 1)

segment_2s_radius_range = 20.0
segment_2s_radius_step = 0.2

# Index is not set to 0 because thickness is applied inward
segment_radius_index = segment_1s_radius_step

maximum_electron_count = 1000
while segment_radius_index <= segment_2s_radius_range:

    inner_radius = segment_radius_index - segment_2s_radius_step
    outer_radius = segment_radius_index

    percentage = lithiumAtom.percent_at_radius(inner_radius, outer_radius)

    electron_count = round(percentage * maximum_electron_count)

    Orbital_2s.create_orbital_segment(
        f"Electron.2s.{segment_radius_index}", outer_radius, electron_count, s2_template
    )

    segment_radius_index = round(segment_radius_index + segment_1s_radius_step, 1)
