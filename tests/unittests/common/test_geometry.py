import carla
import numpy as np
import pytest

from lead.common.geometry import (
    angle_to_yaw_class,
    check_obb_intersection,
    euler_degrees_to_rotation_matrix,
    get_relative_transform,
    normalize_angle_deg,
    normalize_angle_rad,
    rotate_point,
    to_global_frame_2d,
    to_local_frame_2d,
    world_coordinate_2d,
)


def bounding_box(
    location: tuple[float, float, float],
    extent: tuple[float, float, float],
    yaw_deg: float = 0.0,
) -> carla.BoundingBox:
    """Build a CARLA bounding box; the client-side type needs no simulator.

    Args:
        location: Center of the box.
        extent: Half extents of the box.
        yaw_deg: Yaw of the box in degrees.

    Returns:
        The bounding box.
    """
    box = carla.BoundingBox(carla.Location(*location), carla.Vector3D(*extent))
    box.rotation = carla.Rotation(pitch=0.0, yaw=yaw_deg, roll=0.0)
    return box


class TestNormalizeAngleRad:
    """Tests for radian angle normalization."""

    @pytest.mark.parametrize(
        ("angle", "expected"),
        [
            (0.0, 0.0),
            (np.pi / 4, np.pi / 4),
            (-np.pi / 2, -np.pi / 2),
            (1.5 * np.pi, -0.5 * np.pi),
            (2 * np.pi, 0.0),
            (-2 * np.pi, 0.0),
            (4 * np.pi, 0.0),
        ],
    )
    def test_known_values(self, angle, expected):
        """Test normalization against hand-computed values."""
        assert np.isclose(normalize_angle_rad(angle), expected)

    @pytest.mark.parametrize("angle", [np.pi, -np.pi, 3 * np.pi, -3 * np.pi])
    def test_wrap_point_maps_to_positive_pi(self, angle):
        """Test the ±π wrap point resolves to +π, i.e. the range is (-π, π].

        This contradicts the docstring's claim of [-π, π); the guard is ``> π``,
        not ``>= π``. Pinned so a change to either is deliberate.
        """
        assert np.isclose(normalize_angle_rad(angle), np.pi)

    def test_output_always_within_closed_interval(self):
        """Test every output lies in (-π, π] for a dense sweep of inputs."""
        for angle in np.linspace(-10 * np.pi, 10 * np.pi, 501):
            normalized = normalize_angle_rad(float(angle))
            assert -np.pi < normalized <= np.pi + 1e-12

    def test_preserves_angle_modulo_two_pi(self):
        """Test normalization only ever shifts the angle by a multiple of 2π."""
        for angle in np.linspace(-10 * np.pi, 10 * np.pi, 501):
            shift = normalize_angle_rad(float(angle)) - angle
            assert np.isclose(shift % (2 * np.pi), 0.0, atol=1e-9) or np.isclose(
                shift % (2 * np.pi),
                2 * np.pi,
                atol=1e-9,
            )


class TestNormalizeAngleDeg:
    """Tests for degree angle normalization."""

    @pytest.mark.parametrize(
        ("angle", "expected"),
        [
            (0.0, 0.0),
            (90.0, 90.0),
            (450.0, 90.0),
            (-270.0, 90.0),
            (270.0, -90.0),
            (720.0, 0.0),
            (-90.0, -90.0),
        ],
    )
    def test_known_values(self, angle, expected):
        """Test normalization against hand-computed values."""
        assert np.isclose(normalize_angle_deg(angle), expected)

    @pytest.mark.parametrize("angle", [180.0, -180.0, 540.0])
    def test_wrap_point_maps_to_positive_180(self, angle):
        """Test the ±180 wrap point resolves to +180, i.e. the range is (-180, 180].

        The docstring documents ``normalize_angle_deg(180) -> -180.0``, which the
        implementation does not do. Pinned to the implementation.
        """
        assert np.isclose(normalize_angle_deg(angle), 180.0)

    def test_output_always_within_closed_interval(self):
        """Test every output lies in (-180, 180] for a dense sweep of inputs."""
        for angle in np.linspace(-1080.0, 1080.0, 501):
            normalized = normalize_angle_deg(float(angle))
            assert -180.0 < normalized <= 180.0 + 1e-12

    def test_agrees_with_radian_normalization(self):
        """Test the degree and radian variants describe the same wrapping."""
        for angle in np.linspace(-720.0, 720.0, 301):
            from_deg = np.deg2rad(normalize_angle_deg(float(angle)))
            from_rad = normalize_angle_rad(float(np.deg2rad(angle)))
            assert np.isclose(from_deg, from_rad, atol=1e-9)


class TestAngleToYawClass:
    """Tests for the discrete yaw encoding."""

    @pytest.mark.parametrize("num_bins", [4, 12, 16])
    def test_bin_centers_have_zero_residual(self, num_bins):
        """Test angles exactly at a bin center encode with no residual."""
        angle_per_class = 2 * np.pi / num_bins
        for index in range(num_bins):
            angle_cls, angle_res = angle_to_yaw_class(
                index * angle_per_class,
                num_bins,
            )
            assert angle_cls == index
            assert np.isclose(angle_res, 0.0)

    @pytest.mark.parametrize("num_bins", [4, 12, 16])
    def test_reconstruction_round_trip(self, num_bins):
        """Test ``class * width + residual`` recovers the original angle."""
        angle_per_class = 2 * np.pi / num_bins
        for angle in np.linspace(0.0, 2 * np.pi, 97, endpoint=False):
            angle_cls, angle_res = angle_to_yaw_class(float(angle), num_bins)
            reconstructed = angle_cls * angle_per_class + angle_res
            assert np.isclose(reconstructed % (2 * np.pi), angle % (2 * np.pi))

    @pytest.mark.parametrize("num_bins", [4, 12, 16])
    def test_residual_within_half_bin(self, num_bins):
        """Test the residual never exceeds half a bin width."""
        half_width = np.pi / num_bins
        for angle in np.linspace(-4 * np.pi, 4 * np.pi, 401):
            _, angle_res = angle_to_yaw_class(float(angle), num_bins)
            assert -half_width <= angle_res < half_width

    @pytest.mark.parametrize("num_bins", [4, 12, 16])
    def test_class_within_range(self, num_bins):
        """Test the encoded class is always a valid bin index."""
        for angle in np.linspace(-4 * np.pi, 4 * np.pi, 401):
            angle_cls, _ = angle_to_yaw_class(float(angle), num_bins)
            assert 0 <= angle_cls < num_bins

    def test_wraps_to_zero_class_with_negative_residual(self):
        """Test an angle just below 2π wraps into class 0 from below."""
        angle_cls, angle_res = angle_to_yaw_class(2 * np.pi - 1e-9, 12)
        assert angle_cls == 0
        assert angle_res < 0.0

    def test_periodic_in_two_pi(self):
        """Test adding a full turn does not change the encoding."""
        for angle in np.linspace(0.0, 2 * np.pi, 51, endpoint=False):
            base = angle_to_yaw_class(float(angle), 12)
            shifted = angle_to_yaw_class(float(angle) + 2 * np.pi, 12)
            assert base[0] == shifted[0]
            assert np.isclose(base[1], shifted[1])


class TestEulerDegreesToRotationMatrix:
    """Tests for the Euler-to-matrix conversion."""

    def test_identity_at_zero(self):
        """Test all-zero angles give the identity."""
        assert np.allclose(euler_degrees_to_rotation_matrix(0.0, 0.0, 0.0), np.eye(3))

    @pytest.mark.parametrize(
        ("roll", "pitch", "yaw"),
        [(0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (0.0, 45.0, 0.0), (10.0, -20.0, 75.0)],
    )
    def test_is_a_rotation(self, roll, pitch, yaw):
        """Test the result is orthonormal with unit determinant."""
        matrix = euler_degrees_to_rotation_matrix(roll, pitch, yaw)
        assert np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(matrix), 1.0)

    @pytest.mark.parametrize("yaw", [0.0, 30.0, 90.0, -45.0, 200.0])
    def test_pure_yaw_matches_2d_rotation(self, yaw):
        """Test the yaw-only block matches the 2D rotation used elsewhere."""
        matrix = euler_degrees_to_rotation_matrix(0.0, 0.0, yaw)
        yaw_rad = np.deg2rad(yaw)
        expected = np.array(
            [
                [np.cos(yaw_rad), -np.sin(yaw_rad)],
                [np.sin(yaw_rad), np.cos(yaw_rad)],
            ],
        )
        assert np.allclose(matrix[:2, :2], expected)
        assert np.allclose(matrix[2, :], [0.0, 0.0, 1.0])

    def test_composition_order_is_yaw_pitch_roll(self):
        """Test the matrix is the Rz @ Ry @ Rx product, in that order."""
        roll, pitch, yaw = 15.0, -25.0, 40.0
        rx = euler_degrees_to_rotation_matrix(roll, 0.0, 0.0)
        ry = euler_degrees_to_rotation_matrix(0.0, pitch, 0.0)
        rz = euler_degrees_to_rotation_matrix(0.0, 0.0, yaw)
        assert np.allclose(
            euler_degrees_to_rotation_matrix(roll, pitch, yaw),
            rz @ ry @ rx,
        )


class TestFrameTransforms2D:
    """Tests for the 2D local/global frame transforms."""

    def test_identity_transform(self):
        """Test a frame at the origin with zero yaw leaves points alone."""
        point = np.array([3.0, -4.0])
        origin = np.array([0.0, 0.0])
        assert np.allclose(to_local_frame_2d(point, origin, 0.0), point)
        assert np.allclose(to_global_frame_2d(point, origin, 0.0), point)

    def test_translation_only(self):
        """Test a zero-yaw frame reduces to a translation."""
        point = np.array([3.0, -4.0])
        translation = np.array([1.0, 2.0])
        assert np.allclose(
            to_local_frame_2d(point, translation, 0.0),
            point - translation,
        )
        assert np.allclose(
            to_global_frame_2d(point, translation, 0.0),
            point + translation,
        )

    def test_rotation_only(self):
        """Test a 90° frame at the origin rotates as expected."""
        origin = np.array([0.0, 0.0])
        assert np.allclose(
            to_global_frame_2d(np.array([1.0, 0.0]), origin, np.pi / 2),
            [0.0, 1.0],
        )
        assert np.allclose(
            to_local_frame_2d(np.array([0.0, 1.0]), origin, np.pi / 2),
            [1.0, 0.0],
        )

    def test_round_trip(self):
        """Test the two transforms invert each other."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            point = rng.normal(size=2) * 10.0
            translation = rng.normal(size=2) * 10.0
            yaw = float(rng.uniform(-np.pi, np.pi))
            local = to_local_frame_2d(point, translation, yaw)
            assert np.allclose(to_global_frame_2d(local, translation, yaw), point)

    def test_rotation_preserves_relative_distance(self):
        """Test the transform is rigid: distances to the frame origin survive."""
        rng = np.random.default_rng(1)
        for _ in range(20):
            point = rng.normal(size=2) * 10.0
            translation = rng.normal(size=2) * 10.0
            yaw = float(rng.uniform(-np.pi, np.pi))
            local = to_local_frame_2d(point, translation, yaw)
            assert np.isclose(
                np.linalg.norm(local),
                np.linalg.norm(point - translation),
            )


class TestGetRelativeTransform:
    """Tests for the world-to-ego position transform."""

    def test_identity_ego_returns_vehicle_translation(self):
        """Test an ego at the identity returns the other actor's position."""
        ego = np.eye(4)
        vehicle = np.eye(4)
        vehicle[:3, 3] = [5.0, -2.0, 1.0]
        assert np.allclose(get_relative_transform(ego, vehicle), [5.0, -2.0, 1.0])

    def test_same_pose_is_origin(self):
        """Test an actor at the ego's own pose maps to the origin."""
        ego = np.eye(4)
        ego[:3, :3] = euler_degrees_to_rotation_matrix(0.0, 0.0, 33.0)
        ego[:3, 3] = [4.0, 5.0, 6.0]
        assert np.allclose(get_relative_transform(ego, ego), np.zeros(3), atol=1e-12)

    def test_invariant_under_common_transform(self):
        """Test applying one rigid transform to both poses changes nothing."""
        ego = np.eye(4)
        ego[:3, :3] = euler_degrees_to_rotation_matrix(0.0, 0.0, 20.0)
        ego[:3, 3] = [1.0, 2.0, 0.0]
        vehicle = np.eye(4)
        vehicle[:3, :3] = euler_degrees_to_rotation_matrix(0.0, 0.0, -50.0)
        vehicle[:3, 3] = [7.0, -3.0, 0.5]

        common = np.eye(4)
        common[:3, :3] = euler_degrees_to_rotation_matrix(5.0, 10.0, 65.0)
        common[:3, 3] = [-8.0, 4.0, 2.0]

        assert np.allclose(
            get_relative_transform(ego, vehicle),
            get_relative_transform(common @ ego, common @ vehicle),
        )

    def test_rotation_of_ego_rotates_relative_position(self):
        """Test a 90° ego yaw maps a point ahead onto the ego's left axis."""
        ego = np.eye(4)
        ego[:3, :3] = euler_degrees_to_rotation_matrix(0.0, 0.0, 90.0)
        vehicle = np.eye(4)
        vehicle[:3, 3] = [1.0, 0.0, 0.0]
        assert np.allclose(
            get_relative_transform(ego, vehicle),
            [0.0, -1.0, 0.0],
            atol=1e-12,
        )


class TestRotatePoint:
    """Tests for the 2D point rotation."""

    def test_zero_and_full_turn_are_identity(self):
        """Test 0° and 360° leave the point unchanged."""
        point = carla.Vector3D(3.0, 4.0, 5.0)
        for angle in (0.0, 360.0):
            rotated = rotate_point(point, angle)
            assert np.isclose(rotated.x, 3.0)
            assert np.isclose(rotated.y, 4.0)

    def test_quarter_turn(self):
        """Test 90° maps +x onto +y."""
        rotated = rotate_point(carla.Vector3D(1.0, 0.0, 0.0), 90.0)
        assert np.isclose(rotated.x, 0.0, atol=1e-12)
        assert np.isclose(rotated.y, 1.0)

    def test_half_turn_negates(self):
        """Test 180° negates both planar components."""
        rotated = rotate_point(carla.Vector3D(2.0, -3.0, 1.0), 180.0)
        assert np.isclose(rotated.x, -2.0)
        assert np.isclose(rotated.y, 3.0)

    def test_z_is_preserved(self):
        """Test the vertical component is never touched."""
        for angle in (0.0, 37.0, 90.0, 180.0, -45.0):
            assert np.isclose(rotate_point(carla.Vector3D(1.0, 2.0, 9.5), angle).z, 9.5)

    def test_composition_adds_angles(self):
        """Test rotating twice equals rotating by the summed angle."""
        point = carla.Vector3D(1.5, -2.5, 0.0)
        twice = rotate_point(rotate_point(point, 30.0), 45.0)
        once = rotate_point(point, 75.0)
        assert np.isclose(twice.x, once.x)
        assert np.isclose(twice.y, once.y)

    def test_preserves_norm(self):
        """Test rotation is length preserving."""
        point = carla.Vector3D(3.0, 4.0, 0.0)
        rotated = rotate_point(point, 57.0)
        assert np.isclose(np.hypot(rotated.x, rotated.y), 5.0)


class TestWorldCoordinate2D:
    """Tests for the yaw-only local-to-world helper."""

    def test_identity_transform(self):
        """Test an ego at the origin leaves the local point unchanged."""
        transform = carla.Transform(carla.Location(0.0, 0.0, 0.0), carla.Rotation())
        world = world_coordinate_2d(transform, carla.Location(2.0, 3.0, 1.0))
        assert np.allclose([world.x, world.y, world.z], [2.0, 3.0, 1.0])

    def test_translation_and_yaw(self):
        """Test a 90° yaw rotates the local offset before translating."""
        transform = carla.Transform(
            carla.Location(10.0, 20.0, 0.0),
            carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0),
        )
        world = world_coordinate_2d(transform, carla.Location(1.0, 0.0, 0.0))
        assert np.allclose([world.x, world.y], [10.0, 21.0], atol=1e-6)

    def test_pitch_and_roll_are_ignored(self):
        """Test only yaw affects the result, as the docstring promises."""
        location = carla.Location(3.0, -1.0, 0.5)
        flat = carla.Transform(
            carla.Location(1.0, 2.0, 3.0),
            carla.Rotation(pitch=0.0, yaw=25.0, roll=0.0),
        )
        tilted = carla.Transform(
            carla.Location(1.0, 2.0, 3.0),
            carla.Rotation(pitch=15.0, yaw=25.0, roll=-30.0),
        )
        from_flat = world_coordinate_2d(flat, location)
        from_tilted = world_coordinate_2d(tilted, location)
        assert np.allclose(
            [from_flat.x, from_flat.y, from_flat.z],
            [from_tilted.x, from_tilted.y, from_tilted.z],
        )

    def test_vertical_offset_is_added(self):
        """Test the local z is added to the ego z."""
        transform = carla.Transform(carla.Location(0.0, 0.0, 5.0), carla.Rotation())
        world = world_coordinate_2d(transform, carla.Location(0.0, 0.0, 2.0))
        assert np.isclose(world.z, 7.0)


class TestCheckObbIntersection:
    """Tests for the oriented-bounding-box separating-axis check."""

    def test_box_intersects_itself(self):
        """Test a box always overlaps a copy of itself."""
        box = bounding_box((0.0, 0.0, 0.0), (2.0, 1.0, 0.75), yaw_deg=37.0)
        assert check_obb_intersection(box, box)

    def test_overlapping_axis_aligned_boxes(self):
        """Test two clearly overlapping axis-aligned boxes intersect."""
        box_a = bounding_box((0.0, 0.0, 0.0), (2.0, 1.0, 1.0))
        box_b = bounding_box((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
        assert check_obb_intersection(box_a, box_b)

    def test_disjoint_boxes_along_each_axis(self):
        """Test separation along any single axis is detected."""
        box_a = bounding_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        for offset in ((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)):
            assert not check_obb_intersection(
                box_a,
                bounding_box(offset, (1.0, 1.0, 1.0)),
            )

    def test_far_apart_boxes_take_the_bounding_sphere_early_out(self):
        """Test boxes beyond the summed bounding-sphere radius never intersect."""
        box_a = bounding_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), yaw_deg=30.0)
        box_b = bounding_box((100.0, 100.0, 0.0), (1.0, 1.0, 1.0), yaw_deg=-15.0)
        assert not check_obb_intersection(box_a, box_b)

    def test_separated_by_a_face_axis_despite_close_centers(self):
        """Test a rotated pair inside each other's bounding spheres but disjoint.

        The centers are closer than the summed bounding-sphere radii, so the
        early-out does not fire and the separating-axis test must decide.
        """
        box_a = bounding_box((0.0, 0.0, 0.0), (4.0, 0.2, 0.2))
        box_b = bounding_box((0.0, 3.0, 0.0), (4.0, 0.2, 0.2))
        relative = np.array([0.0, 3.0, 0.0])
        reach = np.linalg.norm([4.0, 0.2, 0.2]) * 2
        assert np.linalg.norm(relative) < reach, "early-out would have fired"
        assert not check_obb_intersection(box_a, box_b)

    def test_rotation_can_create_an_overlap(self):
        """Test a box that misses when axis-aligned overlaps once rotated."""
        box_a = bounding_box((0.0, 0.0, 0.0), (4.0, 0.2, 1.0))
        apart = bounding_box((0.0, 3.0, 0.0), (4.0, 0.2, 1.0))
        crossing = bounding_box((0.0, 3.0, 0.0), (4.0, 0.2, 1.0), yaw_deg=90.0)
        assert not check_obb_intersection(box_a, apart)
        assert check_obb_intersection(box_a, crossing)

    def test_is_symmetric(self):
        """Test the check gives the same answer with the arguments swapped."""
        pairs = [
            (
                bounding_box((0.0, 0.0, 0.0), (2.0, 1.0, 1.0), yaw_deg=25.0),
                bounding_box((2.5, 1.0, 0.0), (1.0, 3.0, 1.0), yaw_deg=-40.0),
            ),
            (
                bounding_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                bounding_box((10.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            ),
        ]
        for box_a, box_b in pairs:
            assert check_obb_intersection(box_a, box_b) == check_obb_intersection(
                box_b,
                box_a,
            )

    def test_touching_boxes_count_as_intersecting(self):
        """Test boxes that exactly share a face are reported as overlapping."""
        box_a = bounding_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        box_b = bounding_box((2.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert check_obb_intersection(box_a, box_b)
