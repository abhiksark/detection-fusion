"""Tests for strategy parameter validation."""

import pytest

from detection_fusion.strategies.params import (
    CLUSTERING_SCHEMA,
    NMS_SCHEMA,
    VOTING_SCHEMA,
    ParamSchema,
    ParamSpec,
    get_schema_for_category,
)


class TestParamSpec:
    """Tests for ParamSpec class."""

    def test_validate_float_value(self):
        """Test float value validation."""
        spec = ParamSpec(
            name="threshold",
            param_type="float",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
        )

        assert spec.validate_value(0.7) == 0.7
        assert spec.validate_value("0.8") == 0.8  # String coercion

    def test_validate_int_value(self):
        """Test int value validation."""
        spec = ParamSpec(
            name="count",
            param_type="int",
            default=2,
            min_value=1,
        )

        assert spec.validate_value(3) == 3
        assert spec.validate_value("5") == 5  # String coercion

    def test_validate_bool_value(self):
        """Test bool value validation."""
        spec = ParamSpec(
            name="enabled",
            param_type="bool",
            default=True,
        )

        assert spec.validate_value(True) is True
        assert spec.validate_value(False) is False

    def test_validate_none_returns_default(self):
        """Test None returns default value."""
        spec = ParamSpec(
            name="threshold",
            param_type="float",
            default=0.5,
        )

        assert spec.validate_value(None) == 0.5

    def test_validate_none_required_raises(self):
        """Test None for required parameter raises."""
        spec = ParamSpec(
            name="threshold",
            param_type="float",
            required=True,
        )

        with pytest.raises(ValueError) as excinfo:
            spec.validate_value(None)
        assert "required" in str(excinfo.value)

    def test_validate_min_value(self):
        """Test minimum value validation."""
        spec = ParamSpec(
            name="threshold",
            param_type="float",
            min_value=0.0,
        )

        with pytest.raises(ValueError) as excinfo:
            spec.validate_value(-0.1)
        assert ">=" in str(excinfo.value)

    def test_validate_max_value(self):
        """Test maximum value validation."""
        spec = ParamSpec(
            name="threshold",
            param_type="float",
            max_value=1.0,
        )

        with pytest.raises(ValueError) as excinfo:
            spec.validate_value(1.5)
        assert "<=" in str(excinfo.value)

    def test_validate_choices(self):
        """Test choice validation."""
        spec = ParamSpec(
            name="method",
            param_type="str",
            choices=["iou", "giou", "diou"],
        )

        assert spec.validate_value("iou") == "iou"

        with pytest.raises(ValueError) as excinfo:
            spec.validate_value("unknown")
        assert "one of" in str(excinfo.value)


class TestParamSchema:
    """Tests for ParamSchema class."""

    def test_validate_valid_params(self):
        """Test validating valid parameters."""
        schema = ParamSchema(
            params=[
                ParamSpec(name="iou_threshold", param_type="float", default=0.5),
                ParamSpec(name="min_votes", param_type="int", default=2),
            ]
        )

        result = schema.validate({"iou_threshold": 0.6, "min_votes": 3})
        assert result["iou_threshold"] == 0.6
        assert result["min_votes"] == 3

    def test_validate_fills_defaults(self):
        """Test that missing params get defaults."""
        schema = ParamSchema(
            params=[
                ParamSpec(name="iou_threshold", param_type="float", default=0.5),
                ParamSpec(name="min_votes", param_type="int", default=2),
            ]
        )

        result = schema.validate({"iou_threshold": 0.7})
        assert result["iou_threshold"] == 0.7
        assert result["min_votes"] == 2  # Default

    def test_validate_unknown_params_raises(self):
        """Test unknown parameters raise error."""
        schema = ParamSchema(
            params=[
                ParamSpec(name="iou_threshold", param_type="float", default=0.5),
            ]
        )

        with pytest.raises(ValueError) as excinfo:
            schema.validate({"iou_threshold": 0.5, "unknown_param": 123})
        assert "Unknown parameters" in str(excinfo.value)

    def test_validate_allows_config_param(self):
        """Test that 'config' param is allowed."""
        schema = ParamSchema(
            params=[
                ParamSpec(name="iou_threshold", param_type="float", default=0.5),
            ]
        )

        # Should not raise
        result = schema.validate({"iou_threshold": 0.5, "config": None})
        assert result["iou_threshold"] == 0.5

    def test_get_defaults(self):
        """Test getting all default values."""
        schema = ParamSchema(
            params=[
                ParamSpec(name="iou_threshold", param_type="float", default=0.5),
                ParamSpec(name="min_votes", param_type="int", default=2),
                ParamSpec(name="enabled", param_type="bool", default=True),
            ]
        )

        defaults = schema.get_defaults()
        assert defaults == {
            "iou_threshold": 0.5,
            "min_votes": 2,
            "enabled": True,
        }

    def test_to_dict(self):
        """Test schema serialization."""
        schema = ParamSchema(
            params=[
                ParamSpec(
                    name="threshold",
                    param_type="float",
                    default=0.5,
                    min_value=0.0,
                    max_value=1.0,
                    description="Test param",
                ),
            ]
        )

        result = schema.to_dict()
        assert len(result["params"]) == 1
        assert result["params"][0]["name"] == "threshold"
        assert result["params"][0]["type"] == "float"


class TestPredefinedSchemas:
    """Tests for predefined category schemas."""

    def test_voting_schema(self):
        """Test voting schema has expected params."""
        defaults = VOTING_SCHEMA.get_defaults()
        assert "iou_threshold" in defaults
        assert "min_votes" in defaults

    def test_nms_schema(self):
        """Test NMS schema has expected params."""
        defaults = NMS_SCHEMA.get_defaults()
        assert "iou_threshold" in defaults
        assert "confidence_threshold" in defaults

    def test_clustering_schema(self):
        """Test clustering schema has expected params."""
        defaults = CLUSTERING_SCHEMA.get_defaults()
        assert "eps" in defaults
        assert "min_samples" in defaults

    def test_get_schema_for_category(self):
        """Test getting schema by category name."""
        voting = get_schema_for_category("voting")
        assert voting is VOTING_SCHEMA

        nms = get_schema_for_category("nms")
        assert nms is NMS_SCHEMA

    def test_get_schema_unknown_category(self):
        """Test unknown category returns empty schema."""
        schema = get_schema_for_category("unknown_category")
        assert len(schema.params) == 0


class TestStrategyParamsSchema:
    """Tests for strategy-level params_schema."""

    def test_voting_strategy_has_schema(self):
        """Test MajorityVoting has params_schema."""
        from detection_fusion.strategies.voting import MajorityVoting

        assert MajorityVoting.metadata is not None
        assert MajorityVoting.metadata.params_schema is not None

    def test_voting_strategy_validate_params(self):
        """Test validating params through strategy."""
        from detection_fusion.strategies.voting import MajorityVoting

        strategy = MajorityVoting()
        validated = strategy.validate_params(iou_threshold=0.6, min_votes=3)

        assert validated["iou_threshold"] == 0.6
        assert validated["min_votes"] == 3

    def test_voting_strategy_invalid_params(self):
        """Test invalid params raise error."""
        from detection_fusion.strategies.voting import MajorityVoting

        strategy = MajorityVoting()

        with pytest.raises(ValueError):
            strategy.validate_params(iou_threshold=1.5)  # Out of range

    def test_get_params_schema_classmethod(self):
        """Test get_params_schema class method."""
        from detection_fusion.strategies.voting import MajorityVoting

        schema = MajorityVoting.get_params_schema()
        assert schema is not None
        assert len(schema.params) > 0

    def test_get_param_defaults_classmethod(self):
        """Test get_param_defaults class method."""
        from detection_fusion.strategies.voting import MajorityVoting

        defaults = MajorityVoting.get_param_defaults()
        assert "iou_threshold" in defaults
        assert "min_votes" in defaults
