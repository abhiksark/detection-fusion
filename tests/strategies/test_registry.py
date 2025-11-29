"""Tests for strategy registry."""

import pytest

from detection_fusion.strategies.base import BaseStrategy
from detection_fusion.strategies.registry import (
    StrategyRegistry,
    create_strategy,
    list_strategies,
)


class TestStrategyRegistry:
    """Tests for StrategyRegistry class."""

    def test_list_all_returns_strategies(self):
        """Test list_all returns registered strategies."""
        strategies = StrategyRegistry.list_all()
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert "weighted_vote" in strategies

    def test_list_all_sorted(self):
        """Test list_all returns sorted list."""
        strategies = StrategyRegistry.list_all()
        assert strategies == sorted(strategies)

    def test_create_valid_strategy(self):
        """Test creating valid strategy."""
        strategy = StrategyRegistry.create("weighted_vote")
        assert strategy is not None
        assert isinstance(strategy, BaseStrategy)

    def test_create_with_kwargs(self):
        """Test creating strategy with custom kwargs."""
        strategy = StrategyRegistry.create("weighted_vote", iou_threshold=0.7)
        assert strategy.iou_threshold == 0.7

    def test_create_unknown_raises(self):
        """Test creating unknown strategy raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            StrategyRegistry.create("unknown_strategy_xyz")
        assert "Unknown strategy" in str(excinfo.value)

    def test_get_class(self):
        """Test getting strategy class."""
        cls = StrategyRegistry.get_class("weighted_vote")
        assert cls is not None
        assert issubclass(cls, BaseStrategy)

    def test_get_class_unknown_raises(self):
        """Test getting unknown class raises ValueError."""
        with pytest.raises(ValueError):
            StrategyRegistry.get_class("unknown_strategy_xyz")

    def test_is_registered_true(self):
        """Test is_registered returns True for known strategy."""
        assert StrategyRegistry.is_registered("weighted_vote") is True

    def test_is_registered_false(self):
        """Test is_registered returns False for unknown strategy."""
        assert StrategyRegistry.is_registered("unknown_xyz") is False

    def test_count_positive(self):
        """Test count returns positive number."""
        count = StrategyRegistry.count()
        assert count > 0
        assert count >= 10  # We have at least 10 strategies

    def test_list_by_category_voting(self):
        """Test listing voting strategies."""
        voting = StrategyRegistry.list_by_category("voting")
        assert isinstance(voting, list)

    def test_list_by_category_nms(self):
        """Test listing NMS strategies."""
        nms = StrategyRegistry.list_by_category("nms")
        assert isinstance(nms, list)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_strategy_function(self):
        """Test create_strategy convenience function."""
        strategy = create_strategy("weighted_vote")
        assert strategy is not None
        assert isinstance(strategy, BaseStrategy)

    def test_create_strategy_with_kwargs(self):
        """Test create_strategy with kwargs."""
        strategy = create_strategy("nms", iou_threshold=0.6)
        assert strategy.iou_threshold == 0.6

    def test_list_strategies_function(self):
        """Test list_strategies convenience function."""
        strategies = list_strategies()
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert "weighted_vote" in strategies


class TestStrategyExecution:
    """Tests for strategy execution."""

    def test_weighted_vote_merge(self, multi_model_detections):
        """Test weighted_vote strategy merges detections."""
        strategy = create_strategy("weighted_vote")
        result = strategy.merge(multi_model_detections)

        assert isinstance(result, list)
        # Should produce merged detections

    def test_nms_merge(self, multi_model_detections):
        """Test nms strategy merges detections."""
        strategy = create_strategy("nms")
        result = strategy.merge(multi_model_detections)

        assert isinstance(result, list)

    def test_bayesian_merge(self, multi_model_detections):
        """Test bayesian strategy merges detections."""
        strategy = create_strategy("bayesian")
        result = strategy.merge(multi_model_detections)

        assert isinstance(result, list)

    def test_dbscan_merge(self, multi_model_detections):
        """Test dbscan strategy merges detections."""
        strategy = create_strategy("dbscan")
        result = strategy.merge(multi_model_detections)

        assert isinstance(result, list)
