import math

class KinematicsCalculator:
    def __init__(self):
        # person_id -> last_state
        pass


    # POSITION & TRAJECTORY (FOUNDATION)

    def cenroid_postion(self):
        pass

    def bbox_center_position(self):
        pass

    def trajectory_path(self):
        pass

    def trajectory_length(self):
        pass

    def trajectory_curvature(self):
        pass


    # DISPLACEMENT & DISTANCE

    def frame_to_frame_displacement(self):
        pass

    def total_displacement(self):
        pass

    def total_over_n_frames(self):
        pass

    def max_displacement_step(self):
        pass


    # SPEED

    def instant_speed(self):
        pass

    def avergae_speed(self):
        pass

    def peak_speed(self):
        pass

    def speed_variance(self):
        pass

    def speed_spikes(self):
        pass


    # ACCELERATION

    def instant_acceleration(self):
        pass

    def average_acceleration(self):
        pass

    def peak_acceleration(self):
        pass

    def burst_accelerations(self):
        pass

    def acceleration_variance(self):
        pass



    # POSE / JOINT ANGLES

    def elbow_angle(self):
        pass

    def shoulder_angle(self):
        pass

    def knee_angle(self):
        pass

    def hip_angle(self):

        pass

    def angle_velocity(self):
        pass

    def main(self):
        pass

"""
6️⃣ POSE / JOINT ANGLES (YOU DID WELL HERE)

Your angle list is correct, but here’s the full set:

Upper body

elbow_angle

shoulder_angle

wrist_angle

arm_extension_ratio

Lower body

knee_angle

hip_angle

ankle_angle

leg_extension_ratio

✔️ Punching / kicking = extreme joint angles

7️⃣ ANGULAR KINEMATICS (VERY IMPORTANT)

You already started this — expand it.

joint_angular_velocity

joint_angular_acceleration

angle_change_rate

max_angle_velocity

✔️ Fast arm swings → violence indicator

8️⃣ BODY POSTURE & STABILITY (SINGLE PERSON)

Often overlooked, but VERY useful.

body_lean_angle

center_of_mass_shift

posture_instability

stance_width

foot_separation_ratio

✔️ Fighting stance ≠ walking stance

9️⃣ TEMPORAL PATTERNS (SINGLE PERSON OVER TIME)

These make your system much stronger.

motion_consistency

motion_entropy

pause_to_motion_ratio

active_frames_ratio

✔️ Violence = irregular temporal pattern
🔹 B. Multi-person kinematics (BETWEEN PEOPLE)

These are calculated between two or more people.

Core topics:

Inter-person distance

Rate of distance change

Closing speed (approaching fast)

Sudden proximity events

Collision-like movement

Relative velocity vectors

Directional aggression (moving toward another person)

Encroachment duration (staying too close)

⚠️ Violence often shows:

fast closing speed

sharp distance reduction

repeated proximity spikes"""
