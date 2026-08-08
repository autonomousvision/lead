import numpy as np
import pytest
from py123d.geometry import PoseSE2, PoseSE3

from lead.log_reader.view_geometry import (
    _view_rotation_matrix,
    derive_rig_perturbation,
    to_view_frame_boxes,
    to_view_frame_point_cloud,
    to_view_frame_points,
    to_view_frame_yaws,
    view_frame_center_se2,
)


def rig_relative_matrix(
    perturbation_translation_m: float,
    perturbation_rotation_deg: float,
    rear_axle_to_center_longitudinal: float,
) -> np.ndarray:
    """Forward model of the rig perturbation, as an ISO transform at the rear axle.

    The perturbation is a yaw and a lateral shift at the vehicle center, mirrored
    into ISO (y left, yaw counter-clockwise), then conjugated into the rear-axle
    frame the extrinsics are stored in. This is the transform that
    :func:`derive_rig_perturbation` has to invert.

    Args:
        perturbation_translation_m: Rig translation perturbation in meters.
        perturbation_rotation_deg: Rig yaw perturbation in degrees.
        rear_axle_to_center_longitudinal: X offset of the rear axle shift.

    Returns:
        The 4x4 relative transform between the two camera extrinsics.
    """
    yaw_iso = -np.deg2rad(perturbation_rotation_deg)
    cos_yaw, sin_yaw = np.cos(yaw_iso), np.sin(yaw_iso)
    at_center = np.eye(4)
    at_center[:3, :3] = [[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]]
    at_center[:3, 3] = [0.0, -perturbation_translation_m, 0.0]
    shift = np.eye(4)
    shift[:3, 3] = [rear_axle_to_center_longitudinal, 0.0, 0.0]
    return shift @ at_center @ np.linalg.inv(shift)


def random_pose_se3(rng: np.random.Generator) -> PoseSE3:
    """Draw a random SE3 pose with a normalized quaternion.

    Args:
        rng: Random generator.

    Returns:
        The random pose.
    """
    quaternion = rng.normal(size=4)
    quaternion /= np.linalg.norm(quaternion)
    return PoseSE3(*rng.normal(size=3) * 5.0, *quaternion)


class TestToViewFramePoints:
    """Tests for re-expressing 2D points in the view frame."""

    def test_identity_without_perturbation(self):
        """Test the normal view leaves points untouched."""
        points = np.array([[1.0, 2.0], [-3.0, 4.5], [0.0, 0.0]])
        assert np.allclose(to_view_frame_points(points, 0.0, 0.0), points)

    def test_pure_translation_shifts_lateral_axis(self):
        """Test a translation-only perturbation subtracts it from y."""
        points = np.array([[1.0, 2.0], [-3.0, 4.5]])
        shifted = to_view_frame_points(points, 1.5, 0.0)
        assert np.allclose(shifted[:, 0], points[:, 0])
        assert np.allclose(shifted[:, 1], points[:, 1] - 1.5)

    def test_pure_rotation_preserves_norms(self):
        """Test a rotation-only perturbation is length preserving."""
        points = np.array([[1.0, 2.0], [-3.0, 4.5], [7.0, 0.0]])
        rotated = to_view_frame_points(points, 0.0, 22.0)
        assert np.allclose(
            np.linalg.norm(rotated, axis=1),
            np.linalg.norm(points, axis=1),
        )

    def test_known_quarter_turn(self):
        """Test a 90° perturbation maps +x onto -y."""
        rotated = to_view_frame_points(np.array([[1.0, 0.0]]), 0.0, 90.0)
        assert np.allclose(rotated, [[0.0, -1.0]], atol=1e-12)

    def test_does_not_mutate_input(self):
        """Test the caller's array is left alone."""
        points = np.array([[1.0, 2.0]])
        original = points.copy()
        to_view_frame_points(points, 2.0, 30.0)
        assert np.allclose(points, original)


class TestToViewFrameYaws:
    """Tests for re-expressing yaw angles in the view frame."""

    def test_identity_without_perturbation(self):
        """Test the normal view leaves yaws untouched."""
        yaws = np.array([0.0, 0.5, -1.2, 3.0])
        assert np.allclose(to_view_frame_yaws(yaws, 0.0), yaws)

    def test_subtracts_the_perturbation(self):
        """Test the rig yaw is subtracted, in radians."""
        yaws = np.array([0.0, 0.5])
        assert np.allclose(to_view_frame_yaws(yaws, 30.0), yaws - np.deg2rad(30.0))

    def test_output_within_half_open_interval(self):
        """Test the result is wrapped into [-π, π)."""
        yaws = np.linspace(-4 * np.pi, 4 * np.pi, 401)
        wrapped = to_view_frame_yaws(yaws, 17.0)
        assert np.all(wrapped >= -np.pi)
        assert np.all(wrapped < np.pi)

    def test_wrap_point_maps_to_negative_pi(self):
        """Test π wraps to -π, i.e. the interval is [-π, π).

        This is the opposite endpoint convention to
        ``lead.common.geometry.normalize_angle_rad``, which yields (-π, π].
        Pinned so the divergence stays deliberate.
        """
        assert np.allclose(to_view_frame_yaws(np.array([np.pi]), 0.0), [-np.pi])


class TestToViewFrameBoxes:
    """Tests for re-expressing bounding-box dicts in the view frame."""

    def box(self) -> dict:
        """Build a representative CARLA-style box dict.

        Returns:
            The box dict.
        """
        return {
            "position": [3.0, 4.0, 0.75],
            "yaw": 0.4,
            "class": "car",
            "id": 17,
            "extent": [2.1, 0.9, 0.75],
        }

    def test_identity_without_perturbation(self):
        """Test the normal view leaves the geometry unchanged."""
        view_box = to_view_frame_boxes([self.box()], 0.0, 0.0)[0]
        assert np.allclose(view_box["position"], [3.0, 4.0, 0.75])
        assert np.isclose(view_box["yaw"], 0.4)

    def test_height_and_other_fields_pass_through(self):
        """Test z and all non-geometric fields survive the transform."""
        view_box = to_view_frame_boxes([self.box()], 1.5, 20.0)[0]
        assert np.isclose(view_box["position"][2], 0.75)
        assert view_box["class"] == "car"
        assert view_box["id"] == 17
        assert view_box["extent"] == [2.1, 0.9, 0.75]

    def test_does_not_mutate_the_input_dicts(self):
        """Test the source dicts are copied, not edited in place."""
        boxes = [self.box()]
        to_view_frame_boxes(boxes, 1.5, 20.0)
        assert boxes[0]["position"] == [3.0, 4.0, 0.75]
        assert np.isclose(boxes[0]["yaw"], 0.4)

    def test_agrees_with_the_point_and_yaw_helpers(self):
        """Test the box geometry matches calling the helpers directly."""
        box = self.box()
        view_box = to_view_frame_boxes([box], 1.5, 20.0)[0]
        expected_position = to_view_frame_points(
            np.array([box["position"][:2]]),
            1.5,
            20.0,
        )[0]
        expected_yaw = to_view_frame_yaws(np.array([box["yaw"]]), 20.0)[0]
        assert np.allclose(view_box["position"][:2], expected_position)
        assert np.isclose(view_box["yaw"], expected_yaw)

    def test_empty_input(self):
        """Test an empty box list round-trips to an empty list."""
        assert to_view_frame_boxes([], 1.0, 10.0) == []


class TestToViewFramePointCloud:
    """Tests for re-expressing point clouds in the view frame."""

    def test_identity_without_perturbation(self):
        """Test the normal view leaves the cloud unchanged."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        assert np.allclose(to_view_frame_point_cloud(points, 0.0, 0.0), points)

    def test_trailing_columns_are_preserved(self):
        """Test channels beyond xy survive bit-for-bit."""
        points = np.array([[1.0, 2.0, 3.0, 0.25], [4.0, 5.0, 6.0, 0.75]])
        transformed = to_view_frame_point_cloud(points, 1.5, 20.0)
        assert np.array_equal(transformed[:, 2:], points[:, 2:])

    def test_xy_agrees_with_the_point_helper(self):
        """Test the planar block matches to_view_frame_points."""
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        transformed = to_view_frame_point_cloud(points, 1.5, 20.0)
        assert np.allclose(
            transformed[:, :2],
            to_view_frame_points(points[:, :2], 1.5, 20.0),
        )

    def test_does_not_mutate_input(self):
        """Test the caller's cloud is copied before editing."""
        points = np.array([[1.0, 2.0, 3.0]])
        original = points.copy()
        to_view_frame_point_cloud(points, 1.5, 20.0)
        assert np.allclose(points, original)


class TestViewFrameCenterSe2:
    """Tests for the global pose of the view frame origin."""

    def test_identity_without_perturbation(self):
        """Test the normal view keeps the ego center pose."""
        ego = PoseSE2(x=10.0, y=-5.0, yaw=0.3)
        view = view_frame_center_se2(ego, 0.0, 0.0)
        assert np.allclose([view.x, view.y, view.yaw], [10.0, -5.0, 0.3])

    def test_translation_is_applied_along_the_lateral_axis(self):
        """Test a translation-only perturbation shifts right of the heading."""
        ego = PoseSE2(x=0.0, y=0.0, yaw=0.0)
        view = view_frame_center_se2(ego, 2.0, 0.0)
        assert np.allclose([view.x, view.y], [0.0, -2.0])

    def test_translation_follows_the_heading(self):
        """Test the lateral shift rotates with the ego yaw."""
        ego = PoseSE2(x=0.0, y=0.0, yaw=np.pi / 2)
        view = view_frame_center_se2(ego, 2.0, 0.0)
        assert np.allclose([view.x, view.y], [2.0, 0.0], atol=1e-12)

    def test_rotation_is_subtracted(self):
        """Test the rig yaw is subtracted from the ego yaw."""
        ego = PoseSE2(x=1.0, y=2.0, yaw=0.5)
        view = view_frame_center_se2(ego, 0.0, 30.0)
        assert np.isclose(view.yaw, 0.5 - np.deg2rad(30.0))


class TestDeriveRigPerturbation:
    """Tests for recovering the rig perturbation from camera extrinsics."""

    @pytest.mark.parametrize(
        ("translation_m", "rotation_deg", "lever"),
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.3),
            (1.0, 0.0, 0.0),
            (0.0, 15.0, 0.0),
            (-0.8, -12.0, 1.3),
        ],
    )
    def test_round_trip_known_cases(self, translation_m, rotation_deg, lever):
        """Test the derived scalars match the ones used to build the extrinsics."""
        normal = PoseSE3(1.5, 0.0, 1.6, 1.0, 0.0, 0.0, 0.0)
        perturbated = PoseSE3.from_transformation_matrix(
            rig_relative_matrix(translation_m, rotation_deg, lever)
            @ normal.transformation_matrix,
        )
        derived_translation, derived_rotation = derive_rig_perturbation(
            normal,
            perturbated,
            lever,
        )
        assert np.isclose(derived_translation, translation_m, atol=1e-9)
        assert np.isclose(derived_rotation, rotation_deg, atol=1e-9)

    def test_round_trip_over_random_rigs(self):
        """Test the inverse holds for random perturbations and extrinsics.

        Covers the ISO/CARLA mirroring and the rear-axle lever-arm correction
        together, which is the part of the derivation that is easiest to get
        subtly wrong.
        """
        rng = np.random.default_rng(0)
        for _ in range(100):
            translation_m = float(rng.uniform(-2.0, 2.0))
            rotation_deg = float(rng.uniform(-25.0, 25.0))
            lever = float(rng.uniform(0.0, 3.0))
            normal = random_pose_se3(rng)
            perturbated = PoseSE3.from_transformation_matrix(
                rig_relative_matrix(translation_m, rotation_deg, lever)
                @ normal.transformation_matrix,
            )
            derived_translation, derived_rotation = derive_rig_perturbation(
                normal,
                perturbated,
                lever,
            )
            assert np.isclose(derived_translation, translation_m, atol=1e-9)
            assert np.isclose(derived_rotation, rotation_deg, atol=1e-9)

    def test_identical_extrinsics_give_no_perturbation(self):
        """Test the normal rig is recovered as a zero perturbation."""
        pose = PoseSE3(1.5, 0.0, 1.6, 1.0, 0.0, 0.0, 0.0)
        assert derive_rig_perturbation(pose, pose, 1.3) == (0.0, 0.0)

    def test_returns_plain_floats(self):
        """Test the scalars are floats, not numpy types."""
        pose = PoseSE3(1.5, 0.0, 1.6, 1.0, 0.0, 0.0, 0.0)
        derived = derive_rig_perturbation(pose, pose, 0.0)
        assert all(type(value) is float for value in derived)


class TestViewRotationMatrix:
    """Tests for the rig rotation matrix helper."""

    def test_identity_at_zero(self):
        """Test a zero perturbation gives the identity."""
        assert np.allclose(_view_rotation_matrix(0.0), np.eye(2))

    @pytest.mark.parametrize("rotation_deg", [0.0, 15.0, -40.0, 90.0, 180.0])
    def test_is_a_rotation(self, rotation_deg):
        """Test the matrix is orthonormal with unit determinant."""
        matrix = _view_rotation_matrix(rotation_deg)
        assert np.allclose(matrix.T @ matrix, np.eye(2), atol=1e-12)
        assert np.isclose(np.linalg.det(matrix), 1.0)

    @pytest.mark.parametrize("rotation_deg", [15.0, -40.0, 90.0])
    def test_negated_angle_is_the_inverse(self, rotation_deg):
        """Test negating the angle transposes the matrix."""
        assert np.allclose(
            _view_rotation_matrix(-rotation_deg),
            _view_rotation_matrix(rotation_deg).T,
        )
