import bpy
import math
import random

import scipy.integrate as integrate
import numpy as np


# Reimon area width
SEGMENT_THICKNESS = 0.1


# Items to retain from a new project
PROTECTED_ITEMS = [
    "Camera",
    "Light.Lamp.1",
    "Light.Lamp.2",
    "Material",
    "Background.Plane",
]


start_frame = 0
end_frame = 500


class Orbital_1s:
    @staticmethod
    def rho_function(n, z, r):
        # ρ = 2 * Zr/n
        return 2 * z * r / n

    @staticmethod
    def radial_wave_function(z, p):
        # R1s = 2 * Z^(3/2) * e^(-ρ/2)
        return 2 * z ** (3 / 2) * math.exp(-p / 2)

    @staticmethod
    def angular_wave_function():
        # Y1s = 1 * (1/4π)^(1/2)
        return 1 * (1 / 4 * math.pi) ** (1 / 2)

    @staticmethod
    def superposition_wave_function(r, y):
        # ψ1s = R1s * Y1s
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
    def integrand(x, n, z):
        p = Orbital_1s.rho_function(n, z, x)

        radial_wave = Orbital_1s.radial_wave_function(z, p)
        angular_wave = Orbital_1s.angular_wave_function()
        wave_function = Orbital_1s.superposition_wave_function(
            radial_wave, angular_wave
        )

        return Orbital_1s.radial_distribution_function(x, wave_function)

    @staticmethod
    def create_electron_template(name, base_radius, colour):
        # colour is in (r, g, b, a) format as a percentage from 0 to 1

        bpy.ops.mesh.primitive_ico_sphere_add(radius=base_radius / 10, subdivisions=2)

        current_name = bpy.context.active_object.name
        sphere_object = bpy.data.objects[current_name]

        sphere_object.name = name

        bpy.data.materials.new(f"{name}_mesh")
        material = bpy.data.materials[f"{name}_mesh"]

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
        sphere_object.modifiers[f"{name}_solidify"].thickness = SEGMENT_THICKNESS

        # Particle system setup
        pSystem = sphere_object.modifiers.new(
            f"{name}_particles", type="PARTICLE_SYSTEM"
        )

        pSettings = bpy.data.particles.new(f"{name}_particle_settings")

        pSettings.count = count

        pSettings.frame_start = 0
        pSettings.frame_end = 0
        pSettings.lifetime = end_frame

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

        rotation_degree = random.randint(0, 360)

        start_frame_rotation = rotation_degree + start_frame / 25 / outer_radius
        end_frame_rotation = rotation_degree + end_frame / 25 / outer_radius

        # Segment rotation around axis at start frame
        sphere_object.rotation_euler = (
            start_frame_rotation,
            -start_frame_rotation,
            start_frame_rotation,
        )
        sphere_object.keyframe_insert(
            data_path="rotation_euler",
            frame=start_frame,
        )

        # Segment rotation around axis at end frame
        sphere_object.rotation_euler = (
            end_frame_rotation,
            -end_frame_rotation,
            end_frame_rotation,
        )
        sphere_object.keyframe_insert(data_path="rotation_euler", frame=end_frame)

        # Grow into full size within 100 frames
        sphere_object.scale = (0, 0, 0)
        sphere_object.keyframe_insert(data_path="scale", frame=start_frame)

        sphere_object.scale = (1, 1, 1)
        sphere_object.keyframe_insert(data_path="scale", frame=start_frame + 100)

        # Shrink into nothing size within 100 frames
        sphere_object.scale = (1, 1, 1)
        sphere_object.keyframe_insert(data_path="scale", frame=end_frame - 100)

        sphere_object.scale = (0, 0, 0)
        sphere_object.keyframe_insert(data_path="scale", frame=end_frame)
        return sphere_object

    def __init__(self, quantum_number, effective_charge):
        self.quantum_number = quantum_number
        self.charge = effective_charge

        self.cumulative_1s_area, _ = integrate.quad(
            Orbital_1s.integrand, 0, np.inf, args=(self.quantum_number, self.charge)
        )

        self.template_1s = Orbital_1s.create_electron_template(
            "Orbital.1s.Template", 0.1, (0.8, 0.2, 0.2, 1)
        )

    def percent_at_1s_radius(self, inner_radius, outer_radius):

        area_in_bounds = integrate.quad(
            Orbital_1s.integrand,
            inner_radius,
            outer_radius,
            args=(self.quantum_number, self.charge),
        )

        return area_in_bounds[0] / self.cumulative_1s_area


class Orbital_2s(Orbital_1s):
    @staticmethod
    def radial_wave_function(z, p):
        # R2s = (1/2√2) * (2-ρ) * Z^(3/2) * e^(-ρ/2)
        return 1 / 2 * math.sqrt(2) * (2 - p) * z ** (3 / 2) * math.exp(-p / 2)

    @staticmethod
    def integrand(x, n, z):
        p = Orbital_2s.rho_function(n, z, x)

        radial_wave = Orbital_2s.radial_wave_function(z, p)
        angular_wave = Orbital_2s.angular_wave_function()
        wave_function = Orbital_2s.superposition_wave_function(
            radial_wave, angular_wave
        )

        return Orbital_2s.radial_distribution_function(x, wave_function)

    def __init__(self, quantum_number, effective_charge):

        Orbital_1s.__init__(self, quantum_number, effective_charge)

        self.quantum_number = quantum_number
        self.charge = effective_charge

        self.cumulative_2s_area, _ = integrate.quad(
            Orbital_2s.integrand, 0, np.inf, args=(self.quantum_number, self.charge)
        )

        self.template_2s = Orbital_2s.create_electron_template(
            "Orbital.2s.Template", 0.1, (0.2, 0.4, 0.8, 1)
        )

    def percent_at_2s_radius(self, inner_radius, outer_radius):

        area_in_bounds = integrate.quad(
            Orbital_2s.integrand,
            inner_radius,
            outer_radius,
            args=(self.quantum_number, self.charge),
        )

        return area_in_bounds[0] / self.cumulative_2s_area


class Orbital_3s(Orbital_2s):
    @staticmethod
    def radial_wave_function(z, p):
        # R3s = (1/9√3) * (6-6ρ+ρ^2) * Z^(3/2) * e^(-ρ/2)
        return (
            1 / 9 * 3 ** (1 / 3) * (6 - 6 * p + p**2) * z ** (3 / 2) * math.exp(-p / 2)
        )

    @staticmethod
    def integrand(x, n, z):
        p = Orbital_3s.rho_function(n, z, x)

        radial_wave = Orbital_3s.radial_wave_function(z, p)
        angular_wave = Orbital_3s.angular_wave_function()
        wave_function = Orbital_3s.superposition_wave_function(
            radial_wave, angular_wave
        )

        return Orbital_3s.radial_distribution_function(x, wave_function)

    def __init__(self, quantum_number, effective_charge):

        Orbital_2s.__init__(self, quantum_number, effective_charge)

        self.quantum_number = quantum_number
        self.charge = effective_charge

        self.cumulative_3s_area, _ = integrate.quad(
            Orbital_3s.integrand, 0, np.inf, args=(self.quantum_number, self.charge)
        )

        self.template_3s = Orbital_3s.create_electron_template(
            "Orbital.3s.Template", 0.1, (0.6, 0.8, 0.2, 1)
        )

    def percent_at_3s_radius(self, inner_radius, outer_radius):

        area_in_bounds = integrate.quad(
            Orbital_3s.integrand,
            inner_radius,
            outer_radius,
            args=(self.quantum_number, self.charge),
        )

        return area_in_bounds[0] / self.cumulative_3s_area


class S_Orbital_Atom:

    atom_options = (
        ("h", "hydrogen"),
        ("he", "helium"),
        ("li", "lithium"),
        ("be", "beryllium"),
        ("na", "sodium"),
        ("mg", "magnesium"),
    )

    atom_base_data = {
        "h": {"quantum_number": 1, "effective_charge": 1.0},
        "he": {"quantum_number": 1, "effective_charge": 1.34},
        "li": {"quantum_number": 2, "effective_charge": 1.26},
        "be": {"quantum_number": 2, "effective_charge": 1.58},
        "na": {"quantum_number": 3, "effective_charge": 1.64},
        "mg": {"quantum_number": 3, "effective_charge": 2.25},
    }

    quantum_number_radius = {
        1: 8.0,
        2: 10.0,
        3: 15.0,
    }

    camera_lens_depth = {
        8.0: 35,
        10.0: 35,
        15.0: 35,
    }

    def __init__(
        self,
        name: str,
    ):

        atom_name = ""
        for atom in S_Orbital_Atom.atom_options:
            if name.lower() != atom[0].lower() and name.lower() != atom[1].lower():
                continue

            atom_name = atom[0]
            break

        if atom_name == "":
            raise Exception("Invalid atom name")

        atom_data = S_Orbital_Atom.atom_base_data[atom_name]

        self.quantum_number = atom_data["quantum_number"]
        self.effective_charge = atom_data["effective_charge"]

        self.segment_radius_range = S_Orbital_Atom.quantum_number_radius[
            self.quantum_number
        ]

        if atom_data["quantum_number"] == 1:
            self.atom_obj = Orbital_1s(self.quantum_number, self.effective_charge)

        elif atom_data["quantum_number"] == 2:
            self.atom_obj = Orbital_2s(self.quantum_number, self.effective_charge)
        elif atom_data["quantum_number"] == 3:
            self.atom_obj = Orbital_3s(self.quantum_number, self.effective_charge)

    def create_simulation(self):

        segment_radius_step = SEGMENT_THICKNESS

        # Index is not set to 0 because thickness is applied inward
        segment_radius_index = segment_radius_step * 2

        base_electron_count = 750
        while segment_radius_index <= self.segment_radius_range:

            inner_radius = segment_radius_index - segment_radius_step
            outer_radius = segment_radius_index

            segment_radius_index = round(segment_radius_index + segment_radius_step, 2)

            is_lower_orbital = False

            if isinstance(self.atom_obj, Orbital_3s):
                is_lower_orbital = True

                self.create_3s_orbital_segment(
                    inner_radius, outer_radius, base_electron_count * 20
                )

            if segment_radius_index > S_Orbital_Atom.quantum_number_radius[2]:
                continue

            if is_lower_orbital or isinstance(self.atom_obj, Orbital_2s):
                is_lower_orbital = True

                self.create_2s_orbital_segment(
                    inner_radius, outer_radius, base_electron_count * 10
                )

            if segment_radius_index > S_Orbital_Atom.quantum_number_radius[1]:
                continue

            if is_lower_orbital or isinstance(self.atom_obj, Orbital_1s):
                self.create_1s_orbital_segment(
                    inner_radius, outer_radius, base_electron_count
                )

    def create_1s_orbital_segment(self, inner_radius, outer_radius, electron_count):
        percentage = self.atom_obj.percent_at_1s_radius(inner_radius, outer_radius)
        electron_count = round(percentage * electron_count)

        Orbital_1s.create_orbital_segment(
            f"Electron.1s.{inner_radius}",
            outer_radius,
            electron_count,
            self.atom_obj.template_1s,
        )

    def create_2s_orbital_segment(self, inner_radius, outer_radius, electron_count):
        percentage = self.atom_obj.percent_at_2s_radius(inner_radius, outer_radius)
        electron_count = round(percentage * electron_count)

        Orbital_2s.create_orbital_segment(
            f"Electron.2s.{inner_radius}",
            outer_radius,
            electron_count,
            self.atom_obj.template_2s,
        )

    def create_3s_orbital_segment(self, inner_radius, outer_radius, electron_count):
        percentage = self.atom_obj.percent_at_3s_radius(inner_radius, outer_radius)
        electron_count = round(percentage * electron_count)

        Orbital_3s.create_orbital_segment(
            f"Electron.3s.{inner_radius}",
            outer_radius,
            electron_count,
            self.atom_obj.template_3s,
        )


for obj in bpy.data.objects:
    if obj.name not in PROTECTED_ITEMS:
        bpy.data.objects.remove(obj, do_unlink=True)

for material in bpy.data.materials:
    if material.name not in PROTECTED_ITEMS:
        bpy.data.materials.remove(material, do_unlink=True)


bpy.ops.outliner.orphans_purge()

atom = S_Orbital_Atom("na")


sphere_radius = atom.segment_radius_range


camera_angle_extended_pitch = 75
camera_angle_origin = 90 - camera_angle_extended_pitch
camera_angle__origin_rad = math.radians(camera_angle_origin)


# Trig functions of sin = o/h and cos = a/h
camera_opposite_height = sphere_radius
camera_hypotenuse_radius = camera_opposite_height / math.sin(camera_angle__origin_rad)
camera_adjacent_length = camera_hypotenuse_radius * math.cos(camera_angle__origin_rad)


# The triangle calculated above is aligned to the X-Z / Y-Z plane. Looking
# at the X-Y plane, the adjacent value is the radius. Since the camera is
# angled at 45 degrees in a square, it can be split into two 45-45 triangles
# where the hypotenuse was the previously adjacent value. To calculate the
# adjacent or opposite value for the X-Z plane (both are the same), just divide
# the X-Y plane hypotenuse by the square root of 2 (45-45 properties)

camera_position = (
    camera_adjacent_length / 2 ** (1 / 2),
    -camera_adjacent_length / 2 ** (1 / 2),
    camera_opposite_height,
)

# Camera rotation to look down at 68 degrees
camera_rotation = (math.radians(camera_angle_extended_pitch), 0, math.radians(45))


camera = bpy.data.objects.get("Camera")
camera.data.lens = atom.camera_lens_depth[sphere_radius]
camera.location = camera_position
camera.rotation_euler = camera_rotation


atom.create_simulation()
